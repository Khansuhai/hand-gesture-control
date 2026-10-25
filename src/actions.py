"""Mouse and keyboard actions, plus the action names used in config.json.

Action names you can put in "gesture_actions":
    left_click, right_click, middle_click, double_click
    move, scroll, drag          (continuous, used by point / two_fingers / pinch_hold)
    zoom                        (continuous: Ctrl + mouse wheel while the hand moves up/down)
    volume                      (continuous: volume up/down keys while the hand moves up/down)
    rotate                      (continuous: hold a mouse button and move, e.g. middle-drag
                                 to orbit 3D models; button/modifier set in config "rotate")
    pause, resume, toggle_pause (control the app itself)
    key:<name>                  press one key, e.g. key:enter, key:volumeup, key:playpause
    hotkey:<a>+<b>[+<c>]        key combination, e.g. hotkey:alt+tab, hotkey:ctrl+c
    type:<text>                 type some text
    none (or null)              do nothing

Adding a new action: add a function to SIMPLE_ACTIONS (or a new prefix in
ActionExecutor.run) and it becomes usable from config.json.
"""

from __future__ import annotations

import ctypes
import math
import sys
import time

CONTROL_ACTIONS = {"pause", "resume", "toggle_pause"}
CONTINUOUS_ACTIONS = {"move", "scroll", "drag", "zoom", "rotate", "volume"}
SIMPLE_ACTIONS = {"left_click", "right_click", "middle_click", "double_click"}
NO_ACTION = {"none", ""}
WHEEL_DELTA = 120  # Windows: one mouse-wheel notch

_pyautogui = None


def _pg():
    global _pyautogui
    if _pyautogui is None:
        import pyautogui

        pyautogui.PAUSE = 0  # default is 0.1 s after every call, far too slow
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.FAILSAFE = True  # moving the real mouse into a corner aborts
        _pyautogui = pyautogui
    return _pyautogui


def validate_action(spec: str) -> str | None:
    """Return an error message, or None if `spec` is a valid action."""
    if spec in NO_ACTION or spec in SIMPLE_ACTIONS or spec in CONTROL_ACTIONS or spec in CONTINUOUS_ACTIONS:
        return None
    prefix, sep, arg = spec.partition(":")
    if not sep:
        return f"unknown action '{spec}'"
    if prefix in ("key", "hotkey"):
        keys = arg.lower().split("+") if prefix == "hotkey" else [arg.lower()]
        valid = set(_pg().KEYBOARD_KEYS)
        bad = [k for k in keys if k not in valid]
        if bad or not arg:
            return f"unknown key(s) {bad or [arg]} in '{spec}'"
        return None
    if prefix == "type":
        return None if arg else "type: needs some text"
    return f"unknown action prefix '{prefix}:'"


def _os_double_click_time() -> float:
    if sys.platform == "win32":
        try:
            return ctypes.windll.user32.GetDoubleClickTime() / 1000.0
        except Exception:
            pass
    return 0.5


class ActionExecutor:
    """Sends real input (or only logs it in dry-run mode)."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        pg = _pg()
        self.screen_size = tuple(pg.size())
        self._virtual = [self.screen_size[0] / 2, self.screen_size[1] / 2]
        self._buttons_down: set[str] = set()
        self._keys_down: set[str] = set()
        self._last_left_click: tuple[float, int, int] | None = None
        self.os_double_click_time = _os_double_click_time()

    # -- mouse -------------------------------------------------------------

    def position(self) -> tuple[float, float]:
        if self.dry_run:
            return tuple(self._virtual)
        x, y = _pg().position()
        return float(x), float(y)

    def move_to(self, x: float, y: float) -> None:
        w, h = self.screen_size
        xi = int(min(max(round(x), 0), w - 1))
        yi = int(min(max(round(y), 0), h - 1))
        # Never park the cursor exactly in a corner: that is pyautogui's
        # fail-safe position and would stop the app. Only the real mouse
        # should be able to trigger it.
        if xi in (0, w - 1) and yi in (0, h - 1):
            yi = 1 if yi == 0 else h - 2
        self._virtual = [xi, yi]
        if not self.dry_run:
            _pg().moveTo(xi, yi, _pause=False)

    def mouse_down(self, button: str = "left") -> None:
        self._buttons_down.add(button)
        if not self.dry_run:
            _pg().mouseDown(button=button, _pause=False)

    def mouse_up(self, button: str = "left") -> None:
        self._buttons_down.discard(button)
        if not self.dry_run:
            _pg().mouseUp(button=button, _pause=False)

    def scroll(self, units: int, modifier: str | None = None) -> None:
        """Scroll by raw wheel units (120 = one notch; positive = up).

        modifier="ctrl" makes it Ctrl+wheel, which zooms in most apps.
        """
        if not units or self.dry_run:
            return
        pg = _pg()
        if modifier:
            pg.keyDown(modifier, _pause=False)
        try:
            pg.scroll(int(units), _pause=False)
        finally:
            if modifier:
                pg.keyUp(modifier, _pause=False)

    def key_down(self, key: str) -> None:
        self._keys_down.add(key)
        if not self.dry_run:
            _pg().keyDown(key, _pause=False)

    def key_up(self, key: str) -> None:
        self._keys_down.discard(key)
        if not self.dry_run:
            _pg().keyUp(key, _pause=False)

    def release_all(self) -> None:
        for button in list(self._buttons_down):
            try:
                self.mouse_up(button)
            except Exception:
                pass
        for key in list(self._keys_down):
            try:
                self.key_up(key)
            except Exception:
                pass

    # -- named actions ------------------------------------------------------

    def run(self, spec: str) -> str:
        """Perform a configured action. Returns a short description."""
        if spec in NO_ACTION:
            return ""
        pg = _pg()
        live = not self.dry_run
        if spec == "left_click":
            if live:
                pg.click(_pause=False)
            x, y = self.position()
            self._last_left_click = (time.monotonic(), int(x), int(y))
            return "LEFT CLICK"
        if spec == "double_click":
            # If a left click was just sent at this spot, Windows only needs
            # one more click to see a double click. Sending two would make it
            # a triple click.
            x, y = self.position()
            last = self._last_left_click
            completes = (
                last is not None
                and time.monotonic() - last[0] < self.os_double_click_time * 0.9
                and math.hypot(x - last[1], y - last[2]) <= 2
            )
            if live:
                if completes:
                    pg.click(_pause=False)
                else:
                    pg.doubleClick(_pause=False)
            self._last_left_click = None
            return "DOUBLE CLICK"
        if spec == "right_click":
            if live:
                pg.rightClick(_pause=False)
            return "RIGHT CLICK"
        if spec == "middle_click":
            if live:
                pg.middleClick(_pause=False)
            return "MIDDLE CLICK"
        prefix, _, arg = spec.partition(":")
        if prefix == "key":
            if live:
                pg.press(arg.lower(), _pause=False)
            return f"KEY {arg.upper()}"
        if prefix == "hotkey":
            keys = arg.lower().split("+")
            if live:
                pg.hotkey(*keys, _pause=False)
            return f"HOTKEY {arg.upper()}"
        if prefix == "type":
            if live:
                pg.write(arg, _pause=False)
            return f"TYPE '{arg}'"
        raise ValueError(f"Action '{spec}' cannot be run directly")
