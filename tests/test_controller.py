"""Controller timing logic with a fake mouse/keyboard and a fake clock."""

import pytest

from src.config import load_config
from src.controller import Controller
from src.gestures import GestureResult, HandFeatures

FPS = 30
DT = 1.0 / FPS


class FakeExecutor:
    def __init__(self, screen=(1920, 1080)):
        self.screen_size = screen
        self.pos = [screen[0] / 2, screen[1] / 2]
        self.events: list = []
        self.down = False
        self.moves = 0
        self.os_double_click_time = 0.5
        self.buttons: set = set()
        self.keys: set = set()

    def position(self):
        return tuple(self.pos)

    def move_to(self, x, y):
        self.pos = [x, y]
        self.moves += 1

    def mouse_down(self, button="left"):
        self.buttons.add(button)
        self.down = "left" in self.buttons
        self.events.append("mouse_down" if button == "left" else f"mouse_down:{button}")

    def mouse_up(self, button="left"):
        self.buttons.discard(button)
        self.down = "left" in self.buttons
        self.events.append("mouse_up" if button == "left" else f"mouse_up:{button}")

    def key_down(self, key):
        self.keys.add(key)
        self.events.append(f"key_down:{key}")

    def key_up(self, key):
        self.keys.discard(key)
        self.events.append(f"key_up:{key}")

    def scroll(self, units, modifier=None):
        self.events.append(("zoom" if modifier == "ctrl" else "scroll", units))

    def run(self, spec):
        self.events.append(spec)
        return spec.upper()

    def release_all(self):
        for b in list(self.buttons):
            self.mouse_up(b)
        for k in list(self.keys):
            self.key_up(k)

    def actions(self):
        return [e for e in self.events if isinstance(e, str)]

    def scroll_total(self):
        return sum(e[1] for e in self.events if isinstance(e, tuple) and e[0] == "scroll")

    def zoom_events(self):
        return [e[1] for e in self.events if isinstance(e, tuple) and e[0] == "zoom"]


def result(pose, tip=(0.5, 0.45), pinch_ratio=1.0, mid_ratio=1.0, palm=(0.5, 0.6), pinch_point=None):
    f = HandFeatures(
        fingers={n: False for n in ("thumb", "index", "middle", "ring", "pinky")},
        thumb_up=False, hand_size=100.0, pinch_ratio=pinch_ratio, middle_pinch_ratio=mid_ratio,
        index_pinched=pose == "pinch", middle_pinched=pose == "middle_pinch",
        index_pinch_near=pinch_ratio < 0.4, middle_pinch_near=mid_ratio < 0.4,
    )
    return GestureResult(pose=pose, raw_pose=pose, features=f, index_tip=tip,
                         pinch_point=pinch_point or tip, palm_point=palm)


class Sim:
    def __init__(self, **overrides):
        self.cfg, _, _ = load_config(None, overrides or None)
        self.ex = FakeExecutor()
        self.c = Controller(self.cfg, self.ex)
        self.t = 0.0

    def run(self, res, seconds=None, frames=None):
        n = frames if frames is not None else max(1, round(seconds * FPS))
        for _ in range(n):
            self.c.update(res, self.t)
            self.t += DT

    def tap(self, hold=0.15, **kw):
        self.run(result("pinch", pinch_ratio=0.1, **kw), seconds=hold)
        self.run(result("point", **kw), frames=1)


# ---- cursor -----------------------------------------------------------------


def test_point_moves_cursor_without_initial_jump():
    s = Sim()
    s.run(result("point", tip=(0.5, 0.45)), frames=1)
    assert s.ex.pos == [960, 540]  # first frame only anchors
    s.run(result("point", tip=(0.6, 0.45)), seconds=1.0)
    assert s.ex.pos[0] > 1000  # moved right


def test_active_zone_maps_to_full_screen():
    s = Sim()
    z = s.cfg.active_zone
    tl = s.c._map((z.left, z.top))
    br = s.c._map((1 - z.right, 1 - z.bottom))
    assert tuple(tl) == (0, 0)
    assert tuple(br) == (1919, 1079)
    assert tuple(s.c._map((0.0, 0.0))) == (0, 0)  # outside the zone is clamped


def test_cursor_reaches_screen_edge_once_offset_is_absorbed():
    s = Sim()
    z = s.cfg.active_zone
    # Sweep the hand left and right a few times, then park at the right edge.
    for _ in range(3):
        s.run(result("point", tip=(z.left, 0.4)), seconds=0.5)
        s.run(result("point", tip=(1 - z.right, 0.4)), seconds=0.5)
    s.run(result("point", tip=(1 - z.right, 0.4)), seconds=1.0)
    assert s.ex.pos[0] > 1900


def test_cursor_always_moves_with_the_hand_direction():
    s = Sim()
    s.run(result("point", tip=(0.75, 0.4)), frames=1)  # anchors far from the absolute position
    xs = []
    for i in range(30):  # hand sweeps steadily left
        s.run(result("point", tip=(0.75 - i * 0.01, 0.4)), frames=1)
        xs.append(s.ex.pos[0])
    assert all(b <= a + 1e-6 for a, b in zip(xs, xs[1:]))
    assert xs[-1] < xs[0] - 200


def test_cursor_is_smoothed():
    s = Sim()
    s.run(result("point", tip=(0.5, 0.45)), seconds=0.5)
    start = s.ex.pos[0]
    # Tiny alternating jitter of the fingertip...
    for i in range(60):
        s.run(result("point", tip=(0.5 + (0.002 if i % 2 else -0.002), 0.45)), frames=1)
    # ...moves the cursor much less than the raw jitter (~8 px).
    assert abs(s.ex.pos[0] - start) < 3


def test_other_poses_do_not_move_cursor():
    s = Sim()
    s.run(result("point"), seconds=0.2)
    before = list(s.ex.pos)
    s.run(result("fist", tip=(0.8, 0.8)), seconds=0.5)
    assert s.ex.pos == before


def test_cursor_freezes_when_fingers_close_in_for_a_pinch():
    s = Sim()
    s.run(result("point", tip=(0.5, 0.45)), seconds=0.3)
    before = list(s.ex.pos)
    s.run(result("point", tip=(0.55, 0.5), pinch_ratio=0.3), seconds=0.3)  # thumb approaching
    assert s.ex.pos == before


# ---- clicks -------------------------------------------------------------------


def test_quick_pinch_is_one_left_click():
    s = Sim()
    s.run(result("point"), seconds=0.3)
    s.tap()
    s.run(result("point"), seconds=1.0)
    assert s.ex.actions() == ["left_click"]


def test_cursor_does_not_move_during_click():
    s = Sim()
    s.run(result("point"), seconds=0.3)
    before = list(s.ex.pos)
    s.run(result("pinch", tip=(0.7, 0.7), pinch_ratio=0.1), seconds=0.2)
    s.run(result("point", tip=(0.7, 0.7)), frames=1)
    assert s.ex.pos == before
    assert s.ex.actions() == ["left_click"]


def test_two_quick_pinches_are_a_double_click():
    s = Sim()
    s.run(result("point"), seconds=0.3)
    s.tap(hold=0.12)
    s.run(result("point", pinch_ratio=0.5), seconds=0.1)
    s.tap(hold=0.12)
    s.run(result("point"), seconds=1.0)
    assert s.ex.actions() == ["left_click", "double_click"]


def test_two_slow_pinches_are_two_clicks():
    s = Sim()
    s.run(result("point"), seconds=0.3)
    s.tap()
    s.run(result("point"), seconds=1.0)
    s.tap()
    s.run(result("point"), seconds=1.0)
    assert s.ex.actions() == ["left_click", "left_click"]


def test_pinch_hold_drags_and_release_drops():
    s = Sim()
    s.run(result("point"), seconds=0.3)
    s.run(result("pinch", pinch_ratio=0.1, pinch_point=(0.5, 0.45)), seconds=0.5)
    assert s.ex.events == ["mouse_down"]
    assert s.c.dragging
    before = s.ex.pos[0]
    s.run(result("pinch", pinch_ratio=0.1, pinch_point=(0.65, 0.45)), seconds=0.5)
    assert s.ex.pos[0] > before + 50  # cursor follows while dragging
    s.run(result("point"), frames=1)
    assert s.ex.events == ["mouse_down", "mouse_up"]
    assert not s.c.dragging
    assert "left_click" not in s.ex.events


def test_hand_lost_while_dragging_releases_mouse_after_grace():
    s = Sim()
    s.run(result("pinch", pinch_ratio=0.1), seconds=0.6)
    assert s.ex.down
    s.run(None, seconds=0.15)  # one short dropout: keep dragging
    assert s.ex.down
    s.run(None, seconds=0.5)
    assert not s.ex.down
    assert not s.c.dragging


def test_hand_lost_mid_tap_does_not_click():
    s = Sim()
    s.run(result("pinch", pinch_ratio=0.1), seconds=0.2)
    s.run(None, seconds=1.0)
    s.run(result("point"), seconds=0.5)
    assert s.ex.actions() == []


def test_right_click_fires_once_per_pinch():
    s = Sim()
    s.run(result("point"), seconds=0.3)
    s.run(result("middle_pinch", mid_ratio=0.1), seconds=1.0)
    s.run(result("point"), seconds=1.0)
    s.run(result("middle_pinch", mid_ratio=0.1), seconds=0.3)
    assert s.ex.actions() == ["right_click", "right_click"]


def test_cooldown_blocks_accidental_repeat():
    s = Sim()
    s.run(result("middle_pinch", mid_ratio=0.1), frames=2)
    s.run(result("point"), frames=2)  # very brief release (flicker)
    s.run(result("middle_pinch", mid_ratio=0.1), frames=2)
    assert s.ex.actions() == ["right_click"]


# ---- hold gestures ------------------------------------------------------------


def test_thumbs_up_needs_to_be_held():
    s = Sim()
    s.run(result("thumbs_up"), seconds=0.5)
    s.run(result("point"), seconds=0.5)
    assert s.ex.actions() == []
    s.run(result("thumbs_up"), seconds=2.0)  # held long: still only once
    assert s.ex.actions() == ["key:enter"]


def test_fist_pauses_and_palm_resumes():
    s = Sim()
    s.run(result("point"), seconds=0.3)
    s.run(result("fist"), seconds=1.1)
    assert s.c.paused
    before = list(s.ex.pos)
    s.run(result("point", tip=(0.8, 0.3)), seconds=0.5)
    s.tap()
    s.run(result("thumbs_up"), seconds=1.0)
    assert s.ex.pos == before  # nothing works while paused...
    assert s.ex.actions() == []
    s.run(result("open_palm"), seconds=1.1)  # ...except the resume gesture
    assert not s.c.paused
    s.run(result("point", tip=(0.8, 0.3)), seconds=0.2)
    assert s.ex.pos == before  # resuming never makes the cursor jump
    s.run(result("point", tip=(0.6, 0.3)), seconds=0.5)
    assert s.ex.pos[0] < before[0] - 100  # but it follows the hand again


def test_short_fist_does_not_pause():
    s = Sim()
    s.run(result("fist"), seconds=0.6)
    assert not s.c.paused


def test_pausing_while_dragging_releases_button():
    s = Sim()
    s.run(result("pinch", pinch_ratio=0.1), seconds=0.6)
    assert s.ex.down
    s.c.toggle_pause(s.t)  # Ctrl+Alt+P
    assert not s.ex.down
    assert s.c.paused


def test_hold_progress_for_overlay():
    s = Sim()
    s.run(result("fist"), seconds=0.5)
    pose, frac = s.c.hold_progress
    assert pose == "fist" and 0.4 < frac < 0.6


def test_custom_gesture_mapping():
    s = Sim(gesture_actions={"thumbs_up": "hotkey:alt+tab", "pinch": "right_click"},
            timing={"hold_times_s": {"thumbs_up": 0.2}})
    s.run(result("thumbs_up"), seconds=0.3)
    s.run(result("point"), seconds=0.3)
    s.tap()
    s.run(result("point"), seconds=1.0)
    assert s.ex.actions() == ["hotkey:alt+tab", "right_click"]


def test_disabled_gesture_does_nothing():
    s = Sim(gesture_actions={"thumbs_up": None})
    s.run(result("thumbs_up"), seconds=2.0)
    assert s.ex.actions() == []


# ---- scroll -------------------------------------------------------------------


def test_scroll_up_when_hand_moves_up():
    s = Sim()
    s.run(result("two_fingers", palm=(0.5, 0.6)), frames=1)
    s.run(result("two_fingers", palm=(0.5, 0.45)), seconds=0.5)
    assert s.ex.scroll_total() > 0


def test_scroll_down_when_hand_moves_down():
    s = Sim()
    s.run(result("two_fingers", palm=(0.5, 0.5)), frames=1)
    s.run(result("two_fingers", palm=(0.5, 0.65)), seconds=0.5)
    assert s.ex.scroll_total() < 0


def test_small_movements_inside_deadzone_do_not_scroll():
    s = Sim()
    s.run(result("two_fingers", palm=(0.5, 0.5)), frames=1)
    s.run(result("two_fingers", palm=(0.5, 0.51)), seconds=1.0)
    assert s.ex.scroll_total() == 0


def test_scroll_invert_and_drag_mode():
    s = Sim(scroll={"mode": "drag", "invert": True})
    s.run(result("two_fingers", palm=(0.5, 0.6)), frames=1)
    s.run(result("two_fingers", palm=(0.5, 0.5)), frames=1)
    assert s.ex.scroll_total() < 0


# ---- zoom ---------------------------------------------------------------------


def test_three_fingers_up_zooms_in_with_ctrl_wheel():
    s = Sim()
    s.run(result("three_fingers", palm=(0.5, 0.6)), frames=1)
    s.run(result("three_fingers", palm=(0.5, 0.42)), seconds=1.0)
    z = s.ex.zoom_events()
    assert z and all(u > 0 for u in z)
    assert all(u % 120 == 0 for u in z)  # whole notches only
    assert s.ex.scroll_total() == 0  # no plain scrolling mixed in


def test_three_fingers_down_zooms_out():
    s = Sim()
    s.run(result("three_fingers", palm=(0.5, 0.45)), frames=1)
    s.run(result("three_fingers", palm=(0.5, 0.65)), seconds=1.0)
    assert sum(s.ex.zoom_events()) < 0


def test_switching_from_scroll_to_zoom_starts_fresh():
    s = Sim()
    s.run(result("two_fingers", palm=(0.5, 0.6)), frames=1)
    s.run(result("two_fingers", palm=(0.5, 0.4)), seconds=0.3)
    # Switch to zoom with the hand still high: it is the new starting point,
    # so there is no sudden zoom.
    s.run(result("three_fingers", palm=(0.5, 0.4)), seconds=0.5)
    assert s.ex.zoom_events() == []


# ---- rotate -------------------------------------------------------------------


def test_four_fingers_rotate_with_middle_drag():
    s = Sim()
    s.run(result("point"), seconds=0.3)
    s.run(result("four_fingers", palm=(0.5, 0.6)), seconds=0.2)
    assert s.ex.buttons == {"middle"}
    before = s.ex.pos[0]
    s.run(result("four_fingers", palm=(0.62, 0.6)), seconds=0.5)
    assert s.ex.pos[0] > before + 100  # cursor follows the palm while rotating
    s.run(result("point"), frames=1)
    assert s.ex.buttons == set()
    assert s.ex.actions() == ["mouse_down:middle", "mouse_up:middle"]


def test_rotate_with_modifier_and_other_button():
    s = Sim(rotate={"button": "left", "modifier": "shift"})
    s.run(result("four_fingers"), seconds=0.3)
    assert s.ex.keys == {"shift"} and s.ex.buttons == {"left"}
    s.run(result("fist"), frames=1)
    assert s.ex.keys == set() and s.ex.buttons == set()


@pytest.mark.parametrize("how", ["hand_lost", "pause", "hand_switch", "shutdown"])
def test_rotate_always_releases_button(how):
    s = Sim(rotate={"modifier": "shift"})
    s.run(result("four_fingers"), seconds=0.3)
    assert s.ex.buttons == {"middle"}
    if how == "hand_lost":
        s.run(None, seconds=1.0)
    elif how == "pause":
        s.c.toggle_pause(s.t)
    elif how == "hand_switch":
        s.c.hand_changed(s.t)
    else:
        s.c.shutdown()
    assert s.ex.buttons == set() and s.ex.keys == set()
    assert not s.c.rotating


def test_rotate_speed_scales_movement():
    def moved(speed):
        s = Sim(rotate={"speed": speed})
        s.run(result("four_fingers", palm=(0.5, 0.6)), seconds=0.2)
        start = s.ex.pos[0]
        s.run(result("four_fingers", palm=(0.55, 0.6)), seconds=1.0)
        return s.ex.pos[0] - start

    assert moved(2.0) == pytest.approx(2 * moved(1.0), rel=0.05)


def test_switching_hands_while_dragging_releases_mouse():
    s = Sim()
    s.run(result("pinch", pinch_ratio=0.1), seconds=0.6)
    assert s.ex.down
    s.c.hand_changed(s.t)
    assert not s.ex.down
    s.run(result("point"), seconds=0.5)
    assert "left_click" not in s.ex.actions()


def test_shutdown_releases_mouse():
    s = Sim()
    s.run(result("pinch", pinch_ratio=0.1), seconds=0.6)
    s.c.shutdown()
    assert not s.ex.down


@pytest.mark.parametrize("pose", ["point", "fist", "two_fingers", "none"])
def test_no_actions_from_idle_poses(pose):
    s = Sim()
    s.run(result(pose), seconds=0.5)
    assert s.ex.actions() == []
