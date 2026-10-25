"""Global hotkeys (work even when the preview window is not focused).

Ctrl+Alt+Q quits, Ctrl+Alt+P pauses/resumes. Keys are matched by their
virtual-key code, which is reliable on Windows even while Ctrl is held (the
typed character then becomes a control code).
"""

from __future__ import annotations

from typing import Callable

from pynput import keyboard

VK_Q = 0x51
VK_P = 0x50

_CTRL = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
_ALT = {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr}


def _vk(key) -> int | None:
    vk = getattr(key, "vk", None)
    if vk is None and hasattr(key, "value"):
        vk = getattr(key.value, "vk", None)
    return vk


class Hotkeys:
    def __init__(self, on_quit: Callable[[], None], on_toggle_pause: Callable[[], None]):
        self.on_quit = on_quit
        self.on_toggle_pause = on_toggle_pause
        self._ctrl: set = set()
        self._alt: set = set()
        self._held: set[int] = set()
        self._listener: keyboard.Listener | None = None

    def press(self, key) -> None:
        if key in _CTRL:
            self._ctrl.add(key)
            return
        if key in _ALT:
            self._alt.add(key)
            if key == keyboard.Key.alt_gr:  # AltGr == Ctrl+Alt on many layouts
                self._ctrl.add(key)
            return
        vk = _vk(key)
        if vk is None:
            return
        repeat = vk in self._held  # holding a key auto-repeats; act once
        self._held.add(vk)
        if repeat or not (self._ctrl and self._alt):
            return
        if vk == VK_Q:
            self.on_quit()
        elif vk == VK_P:
            self.on_toggle_pause()

    def release(self, key) -> None:
        self._ctrl.discard(key)
        self._alt.discard(key)
        vk = _vk(key)
        if vk is not None:
            self._held.discard(vk)

    def start(self) -> None:
        self._listener = keyboard.Listener(on_press=self.press, on_release=self.release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
