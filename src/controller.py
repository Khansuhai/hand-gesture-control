"""Turns the stream of poses into mouse/keyboard actions.

This is where timing lives: clicks vs. drags, double clicks, hold-to-trigger
gestures, cooldowns, cursor freezing and smoothing. It never talks to
pyautogui directly - it calls an executor (src/actions.py), so it can be
tested with a fake one and a fake clock.
"""

from __future__ import annotations

import math

import numpy as np

from src.actions import CONTINUOUS_ACTIONS, CONTROL_ACTIONS, NO_ACTION, WHEEL_DELTA
from src.gestures import GestureResult
from src.smoothing import OneEuroFilter2D

# Poses with their own logic below; every other pose goes through the
# generic "hold for N seconds, then fire once" path.
_SPECIAL_POSES = {"pinch", "none"}
# kind -> (units per step, send in chunks of, status up, status down).
# Scroll goes in thirds of a notch (smooth); zoom in whole notches because
# many apps zoom one step per wheel event; volume in single key presses.
_WHEEL_KINDS = {
    "scroll": (WHEEL_DELTA, 40, "SCROLL UP", "SCROLL DOWN"),
    "zoom": (WHEEL_DELTA, WHEEL_DELTA, "ZOOM IN", "ZOOM OUT"),
    "volume": (1, 1, "VOLUME UP", "VOLUME DOWN"),
}


def _is_set(action) -> bool:
    return action is not None and action not in NO_ACTION


class Controller:
    def __init__(self, cfg, executor):
        self.base_cfg = cfg
        self.cfg = cfg  # = base_cfg, or the active app profile's config
        self.profile = "default"
        self.ex = executor
        self.screen_w, self.screen_h = executor.screen_size
        s = cfg.smoothing
        self.filter = OneEuroFilter2D(s.min_cutoff, s.beta, s.d_cutoff)
        self.actions: dict = cfg.gesture_actions

        self.two_hand_active = False
        self._two_ref = 0.0
        self._two_acc = 0.0
        self._pinch_blocked = False
        self.paused = False
        self.dragging = False
        self.rotating = False
        self.cursor: np.ndarray | None = None
        self.last_action: tuple[str, float] = ("", -1e9)

        self._offset = np.zeros(2)
        self._moving = False
        self._last_mapped: np.ndarray | None = None
        self._anchor: tuple[np.ndarray, np.ndarray] | None = None
        self._freeze_until = -1e9
        self._last_fire: dict[str, float] = {}
        self._last_seen: float | None = None
        self._last_tap: float | None = None
        self._reset_hand_state()

    # -- public ----------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "PAUSED" if self.paused else "ACTIVE"

    @property
    def hold_progress(self) -> tuple[str, float] | None:
        """(pose, 0..1) while a hold gesture is charging up, for the overlay."""
        if self._hold_pose is None or self._hold_fired or self._hold_time <= 0:
            return None
        frac = (self._hold_now - self._hold_start) / self._hold_time
        return self._hold_pose, max(0.0, min(1.0, frac))

    def update(self, result: GestureResult | None, now: float, two_hand: float | None = None) -> None:
        """One camera frame. `two_hand` = distance between both hands' pinches, if both pinch."""
        if two_hand is not None and not self.paused and _is_set(self.actions.get("two_hand_pinch")):
            self._last_seen = now
            self._update_two_hand(two_hand, now)
            return
        if self.two_hand_active:
            self.two_hand_active = False
            # The hand may still be pinched: ignore that pinch until it opens,
            # so ending a two-hand zoom never clicks.
            self._pinch_blocked = True
            self._moving = False

        if result is None:
            if self._last_seen is not None and now - self._last_seen > self.cfg.timing.hand_lost_grace_s:
                self._release_holds(now, "(hand lost)")
                self._reset_hand_state()
            self._moving = False
            return
        self._last_seen = now
        pose = result.pose

        self._update_hold(pose, now)
        if self.paused:
            self._moving = False
            return
        self._update_pinch(result, now)
        self._update_rotate(result, now)
        self._update_wheel(result, now)
        self._update_cursor(result, now)

    def trigger(self, event: str, now: float) -> bool:
        """Fire a one-off event detected elsewhere (e.g. a swipe)."""
        if self.paused or self.two_hand_active:
            return False
        return self._fire(event, now)

    def set_app(self, exe: str | None, now: float) -> None:
        """Switch to the profile matching the foreground program (or back to default)."""
        name, cfg = "default", self.base_cfg
        exe = (exe or "").lower()
        for pname, prof in self.base_cfg.profiles.items():
            if exe in prof.match:
                name, cfg = pname, prof.cfg
                break
        if name == self.profile:
            return
        self._release_holds(now)  # buttons/keys may differ between profiles
        self.profile = name
        self.cfg = cfg
        self.actions = cfg.gesture_actions
        self._wheel_kind = None
        self._status(f"PROFILE: {name}", now)

    def set_paused(self, paused: bool, now: float) -> None:
        if paused == self.paused:
            return
        if paused:
            self._release_holds(now)
        self.paused = paused
        self._pinch_start = None
        self._wheel_ref = None
        self._moving = False
        self._status("PAUSED" if paused else "RESUMED", now)

    def toggle_pause(self, now: float) -> None:
        self.set_paused(not self.paused, now)

    def hand_changed(self, now: float) -> None:
        """Control moved to the other hand: finish anything the old hand started."""
        self._release_holds(now, "(hand switched)")
        self._reset_hand_state()
        self._last_tap = None

    def shutdown(self) -> None:
        self._release_holds(0.0)
        self.ex.release_all()

    # -- internals ---------------------------------------------------------------

    def _release_holds(self, now: float, why: str = "") -> None:
        """Let go of any mouse button / key a gesture is holding down."""
        if self.dragging:
            self._end_drag(now, f"DROP {why}".strip())
        if self.rotating:
            self._end_rotate(now, f"ROTATE END {why}".strip())

    def _update_two_hand(self, dist: float, now: float) -> None:
        """Both hands pinching: pull apart = zoom in, push together = zoom out."""
        if not self.two_hand_active:
            self._release_holds(now)
            self._pinch_start = None  # cancel a single-hand click in progress
            self._last_tap = None
            self.two_hand_active = True
            self._two_ref = dist
            self._two_acc = 0.0
            self._status("TWO-HAND ZOOM", now)
            return
        th = self.cfg.two_hand
        notches = th.speed * (dist - self._two_ref) / 0.1
        self._two_ref = dist
        if th.invert:
            notches = -notches
        self._two_acc += notches * WHEEL_DELTA
        step = int(self._two_acc / WHEEL_DELTA) * WHEEL_DELTA
        if step:
            self._two_acc -= step
            if self.actions.get("two_hand_pinch") == "zoom":
                self.ex.scroll(step, modifier="ctrl")
            else:
                self.ex.scroll(step)
            self._status("ZOOM IN" if step > 0 else "ZOOM OUT", now)

    def _update_rotate(self, r: GestureResult, now: float) -> None:
        want = self.actions.get(r.pose) == "rotate"
        if want and not self.rotating and not self.dragging:
            rc = self.cfg.rotate
            if rc.modifier != "none":
                self.ex.key_down(rc.modifier)
            self.ex.mouse_down(rc.button)
            self.rotating = True
            self._moving = False  # anchor on the palm without a jump
            self._status("ROTATE", now)
        elif not want and self.rotating:
            self._end_rotate(now, "ROTATE END")

    def _end_rotate(self, now: float, text: str) -> None:
        rc = self.cfg.rotate
        self.ex.mouse_up(rc.button)
        if rc.modifier != "none":
            self.ex.key_up(rc.modifier)
        self.rotating = False
        self._moving = False
        self._status(text, now)

    def _reset_hand_state(self) -> None:
        self._pinch_start: float | None = None
        self._pinch_hold_fired = False
        self._hold_pose: str | None = None
        self._hold_start = 0.0
        self._hold_now = 0.0
        self._hold_time = 0.0
        self._hold_fired = False
        self._wheel_ref: float | None = None
        self._wheel_prev = 0.0
        self._wheel_time = 0.0
        self._wheel_acc = 0.0
        self._wheel_kind: str | None = None
        self._moving = False

    def _status(self, text: str, now: float) -> None:
        if text:
            self.last_action = (text, now)

    def _fire(self, event: str, now: float) -> bool:
        action = self.actions.get(event)
        if not _is_set(action) or action in CONTINUOUS_ACTIONS:
            return False
        t = self.cfg.timing
        cooldown = t.cooldowns_s.get(event, t.default_cooldown_s)
        last = self._last_fire.get(event)
        if last is not None and now - last < cooldown:
            return False
        self._last_fire[event] = now
        if action in CONTROL_ACTIONS:
            if action == "pause":
                self.set_paused(True, now)
            elif action == "resume":
                self.set_paused(False, now)
            else:
                self.toggle_pause(now)
            return True
        self._status(self.ex.run(action), now)
        return True

    def _update_hold(self, pose: str, now: float) -> None:
        """Generic: pose held for hold_times_s[pose] seconds fires once."""
        if pose != self._hold_pose:
            self._hold_pose = pose
            self._hold_start = now
            self._hold_fired = False
        self._hold_now = now
        action = self.actions.get(pose)
        if pose in _SPECIAL_POSES or not _is_set(action) or action in CONTINUOUS_ACTIONS:
            self._hold_time = 0.0
            return
        self._hold_time = float(self.cfg.timing.hold_times_s.get(pose, 0.0))
        if self._hold_fired:
            return
        if self.paused and action not in ("resume", "toggle_pause"):
            return
        if now - self._hold_start >= self._hold_time:
            self._hold_fired = True  # one gesture = one action until the pose changes
            self._fire(pose, now)

    def _update_pinch(self, r: GestureResult, now: float) -> None:
        t = self.cfg.timing
        hold_action = self.actions.get("pinch_hold")
        if self._pinch_blocked:
            if r.pose != "pinch":
                self._pinch_blocked = False
            return
        if r.pose == "pinch":
            if self._pinch_start is None:
                self._pinch_start = now
                self._pinch_hold_fired = False
            elif (
                not self._pinch_hold_fired
                and _is_set(hold_action)
                and now - self._pinch_start >= t.drag_hold_s
            ):
                self._pinch_hold_fired = True
                if hold_action == "drag":
                    self._start_drag(now)
                else:
                    self._fire("pinch_hold", now)
            return

        if self._pinch_start is None:
            return
        # Pinch released.
        if self.dragging:
            self._end_drag(now, "DROP")
        elif not self._pinch_hold_fired:
            double_ok = _is_set(self.actions.get("double_pinch"))
            if double_ok and self._last_tap is not None and now - self._last_tap <= t.double_click_window_s:
                self._fire("double_pinch", now)
                self._last_tap = None
            else:
                self._fire("pinch", now)
                self._last_tap = now
            # Keep the cursor still so a second tap lands on the same spot.
            self._freeze_until = now + t.release_freeze_s
        self._pinch_start = None

    def _start_drag(self, now: float) -> None:
        self.ex.mouse_down()
        self.dragging = True
        self._moving = False  # re-anchor on the pinch point without a jump
        self._status("DRAG", now)

    def _end_drag(self, now: float, text: str) -> None:
        self.ex.mouse_up()
        self.dragging = False
        self._moving = False
        self._status(text, now)

    def _update_wheel(self, r: GestureResult, now: float) -> None:
        """Hand moving up/down -> scroll (wheel), zoom (Ctrl+wheel) or volume keys."""
        kind = self.actions.get(r.pose)
        if kind not in _WHEEL_KINDS or kind != self._wheel_kind:
            self._wheel_ref = None
            self._wheel_acc = 0.0
            self._wheel_kind = kind
            if kind not in _WHEEL_KINDS:
                return
        y = r.palm_point[1]
        if self._wheel_ref is None:
            self._wheel_ref = y
            self._wheel_prev = y
            self._wheel_time = now
            return
        sc = getattr(self.cfg, kind)
        dt = max(0.0, now - self._wheel_time)
        self._wheel_time = now
        if sc.mode == "joystick":
            # Hand above the starting height goes up, below goes down; further
            # away = faster. `speed` = steps/second at 10% of frame height.
            disp = self._wheel_ref - y
            mag = abs(disp) - sc.deadzone
            amount = math.copysign(sc.speed * (mag / 0.1) * dt, disp) if mag > 0 else 0.0
        else:
            # Touchpad style: follows hand movement.
            # `speed` = steps per 10% of frame height moved.
            amount = sc.speed * (self._wheel_prev - y) / 0.1
        self._wheel_prev = y
        if sc.invert:
            amount = -amount

        unit, chunk, up, down = _WHEEL_KINDS[kind]
        self._wheel_acc += amount * unit
        step = int(self._wheel_acc / chunk) * chunk
        if not step:
            return
        self._wheel_acc -= step
        if kind == "volume":
            key = "volumeup" if step > 0 else "volumedown"
            for _ in range(abs(step)):
                self.ex.run(f"key:{key}")
        else:
            self.ex.scroll(step, modifier="ctrl" if kind == "zoom" else None)
        self._status(up if step > 0 else down, now)

    def _map(self, p: tuple[float, float]) -> np.ndarray:
        """Active-zone coordinates (0..1 of the frame) -> screen pixels."""
        z = self.cfg.active_zone
        u = (p[0] - z.left) / (1.0 - z.left - z.right)
        v = (p[1] - z.top) / (1.0 - z.top - z.bottom)
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)
        return np.array([u * (self.screen_w - 1), v * (self.screen_h - 1)])

    def _update_cursor(self, r: GestureResult, now: float) -> None:
        f = r.features
        if self.dragging:
            point = r.pinch_point
        elif self.rotating:
            point = r.palm_point  # the knuckle barely jitters: smooth orbiting
        elif self.actions.get(r.pose) == "move":
            frozen = (
                now < self._freeze_until
                or self._pinch_start is not None
                # Fingers closing in on a pinch: stop before the click so the
                # cursor doesn't slide off the button.
                or f.index_pinch_near
                or f.middle_pinch_near
            )
            if frozen:
                self._moving = False
                return
            point = r.index_tip
        else:
            self._moving = False
            return

        mapped = self._map(point)
        if not self._moving:
            # (Re)start from wherever the cursor is now, so switching control
            # points, unfreezing or a new hand never makes the cursor jump.
            current = np.array(self.ex.position(), dtype=float)
            self._offset = current - mapped
            self.filter.reset((current[0], current[1]), now)
            self._last_mapped = mapped
            self._anchor = (current, mapped)
            self.cursor = current
            self._moving = True
            return

        delta = mapped - self._last_mapped
        self._last_mapped = mapped
        if self.rotating:
            # Relative movement from where the rotation started, scaled by rotate.speed.
            start_cursor, start_mapped = self._anchor
            self._offset = start_cursor + (mapped - start_mapped) * self.cfg.rotate.speed - mapped
        elif not self.dragging:
            # Shrink the offset a little with every movement so hand and
            # cursor line up again (edge of zone = edge of screen). Since the
            # rate is < 1, the cursor always moves the same way as the hand,
            # just a bit faster or slower until they are aligned.
            rate = self.cfg.smoothing.offset_absorb
            for axis in (0, 1):
                shrink = min(abs(self._offset[axis]), rate * abs(delta[axis]))
                self._offset[axis] -= math.copysign(shrink, self._offset[axis])
        target = mapped + self._offset
        target[0] = min(max(target[0], 0.0), self.screen_w - 1)
        target[1] = min(max(target[1], 0.0), self.screen_h - 1)
        x, y = self.filter(float(target[0]), float(target[1]), now)
        self.ex.move_to(x, y)
        self.cursor = np.array([x, y])
