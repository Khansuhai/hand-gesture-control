"""Name of the program whose window is in front (e.g. "chrome.exe").

Used to switch per-app gesture profiles. Windows only; returns None elsewhere
or if the name can't be read (e.g. for some system windows).
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def foreground_exe() -> str | None:
    if sys.platform != "win32":
        return None
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return None
        return os.path.basename(buf.value).lower()
    finally:
        kernel32.CloseHandle(handle)
