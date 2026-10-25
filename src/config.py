"""Load and validate config.json.

The defaults below are the single source of truth: config.json only needs to
contain the values you want to change. Keys starting with "_" are treated as
comments and ignored.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

DEFAULTS: dict[str, Any] = {
    "camera": {
        "index": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "backend": "dshow",  # dshow | msmf | any
        "mirror": True,
    },
    "tracking": {
        "hand": "right",  # right | left | any  (the preferred hand)
        "use_other_hand_if_missing": True,
        "switch_delay_s": 0.3,
        "max_hands": 2,
        "min_detection_confidence": 0.6,
        "min_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "model_path": "models/hand_landmarker.task",
        "model_url": MODEL_URL,
    },
    # Fractions of the camera frame cut off on each side. The rectangle left
    # in the middle is mapped to the whole screen.
    "active_zone": {"left": 0.2, "right": 0.2, "top": 0.12, "bottom": 0.38},
    "smoothing": {
        "min_cutoff": 1.0,
        "beta": 0.005,
        "d_cutoff": 1.0,
        "offset_absorb": 0.4,
    },
    "gestures": {
        "stable_frames": 3,
        "finger_extended_ratio": 1.15,
        "thumb_extended_ratio": 0.5,
        "thumb_up_min_alignment": 0.7,
        "pinch_on_ratio": 0.25,
        "pinch_off_ratio": 0.35,
        "pinch_min_extension": 0.85,
        "pinch_freeze_ratio": 0.4,
    },
    "timing": {
        "drag_hold_s": 0.4,
        "double_click_window_s": 0.4,
        "release_freeze_s": 0.25,
        "hand_lost_grace_s": 0.3,
        "hold_times_s": {
            "thumbs_up": 0.7,
            "fist": 1.0,
            "open_palm": 1.0,
            "thumbs_down": 0.8,
            "rock": 0.6,
            "pinky": 0.8,
        },
        "cooldowns_s": {
            "pinch": 0.15,
            "double_pinch": 0.3,
            "middle_pinch": 0.5,
            "thumbs_up": 1.0,
            "fist": 1.0,
            "open_palm": 1.0,
            "thumbs_down": 1.5,
            "rock": 1.0,
            "pinky": 1.5,
            "swipe_left": 0.6,
            "swipe_right": 0.6,
            "swipe_up": 0.8,
            "swipe_down": 0.8,
        },
        "default_cooldown_s": 0.3,
    },
    "scroll": {
        "mode": "joystick",  # joystick | drag
        "speed": 8.0,
        "deadzone": 0.03,
        "invert": False,
    },
    # Ctrl + mouse wheel (zoom in browsers, Word, photo viewers, many 3D apps).
    "zoom": {
        "mode": "joystick",  # joystick | drag
        "speed": 3.0,
        "deadzone": 0.03,
        "invert": False,
    },
    # Hold a mouse button and move the hand: orbit/rotate in 3D apps.
    # middle = Blender, SketchUp, Google Earth (tilt/rotate); Fusion 360 uses
    # middle + shift; many web 3D viewers use left.
    "rotate": {
        "button": "middle",  # left | middle | right
        "modifier": "none",  # none | shift | ctrl | alt
        "speed": 1.0,
    },
    # Volume keys while the hand moves up/down (🤙 call-me sign by default).
    "volume": {
        "mode": "joystick",  # joystick | drag
        "speed": 10.0,  # volume steps per second at 10% of frame height
        "deadzone": 0.03,
        "invert": False,
    },
    # Quick open-palm swipes.
    "swipe": {
        "min_distance": 0.22,  # fraction of the frame the palm must travel
        "max_time_s": 0.35,  # ...within this time
        "cooldown_s": 0.6,
    },
    # Pinch with both hands and pull apart / push together.
    "two_hand": {
        "speed": 4.0,  # wheel notches per 10% of frame width
        "invert": False,
    },
    # Event/pose name -> action. See actions.py for the list of actions.
    "gesture_actions": {
        "point": "move",
        "two_fingers": "scroll",
        "three_fingers": "zoom",
        "four_fingers": "rotate",
        "call_me": "volume",
        "two_hand_pinch": "zoom",
        "pinch": "left_click",
        "double_pinch": "double_click",
        "pinch_hold": "drag",
        "middle_pinch": "right_click",
        "thumbs_up": "key:enter",
        "fist": "pause",
        "open_palm": "resume",
        "swipe_left": "hotkey:alt+shift+tab",
        "swipe_right": "hotkey:alt+tab",
        "swipe_up": "hotkey:win+tab",
        "swipe_down": "hotkey:win+d",
        "rock": "key:playpause",
        "thumbs_down": "hotkey:win+ctrl+o",
        "pinky": "hotkey:win+shift+s",
    },
    # Per-app profiles: when the foreground app's .exe matches, these settings
    # replace the ones above. Any of gesture_actions, scroll, zoom, rotate,
    # volume, two_hand, swipe can be overridden.
    "profiles": {
        "browser": {
            "match": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe"],
            "gesture_actions": {"swipe_left": "hotkey:alt+left", "swipe_right": "hotkey:alt+right"},
        },
        "powerpoint": {
            "match": ["powerpnt.exe"],
            "gesture_actions": {"swipe_left": "key:left", "swipe_right": "key:right"},
        },
        "google_earth": {
            # Google Earth zooms with the plain mouse wheel; arrows move the view.
            "match": ["googleearth.exe"],
            "gesture_actions": {
                "three_fingers": "scroll",
                "two_hand_pinch": "scroll",
                "swipe_left": "key:left",
                "swipe_right": "key:right",
            },
        },
        "blender": {
            # Blender: wheel = zoom, middle-drag = orbit.
            "match": ["blender.exe"],
            "gesture_actions": {"three_fingers": "scroll", "two_hand_pinch": "scroll"},
        },
    },
    "ui": {"preview": True, "window_scale": 1.0, "show_debug": True},
}

# Sections whose keys are user-defined (any gesture name is allowed).
_FREE_FORM = {("gesture_actions",), ("timing", "hold_times_s"), ("timing", "cooldowns_s"), ("profiles",)}
PROFILE_SECTIONS = ("gesture_actions", "scroll", "zoom", "rotate", "volume", "two_hand", "swipe")

_RANGES: dict[tuple[str, ...], tuple[float, float]] = {
    ("camera", "index"): (0, 20),
    ("camera", "width"): (160, 3840),
    ("camera", "height"): (120, 2160),
    ("camera", "fps"): (5, 120),
    ("tracking", "max_hands"): (1, 4),
    ("tracking", "switch_delay_s"): (0.0, 5.0),
    ("tracking", "min_detection_confidence"): (0.0, 1.0),
    ("tracking", "min_presence_confidence"): (0.0, 1.0),
    ("tracking", "min_tracking_confidence"): (0.0, 1.0),
    ("active_zone", "left"): (0.0, 0.45),
    ("active_zone", "right"): (0.0, 0.45),
    ("active_zone", "top"): (0.0, 0.45),
    ("active_zone", "bottom"): (0.0, 0.45),
    ("smoothing", "offset_absorb"): (0.0, 0.9),
    ("rotate", "speed"): (0.1, 5.0),
    ("swipe", "min_distance"): (0.05, 0.8),
    ("swipe", "max_time_s"): (0.1, 2.0),
    ("gestures", "stable_frames"): (1, 15),
    ("ui", "window_scale"): (0.25, 3.0),
}

_CHOICES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("camera", "backend"): ("dshow", "msmf", "any"),
    ("tracking", "hand"): ("right", "left", "any"),
    ("scroll", "mode"): ("joystick", "drag"),
    ("zoom", "mode"): ("joystick", "drag"),
    ("volume", "mode"): ("joystick", "drag"),
    ("rotate", "button"): ("left", "middle", "right"),
    ("rotate", "modifier"): ("none", "shift", "ctrl", "alt"),
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict, path: tuple[str, ...], warnings: list[str]) -> dict:
    out = copy.deepcopy(base)
    free_form = path in _FREE_FORM
    for key, value in override.items():
        if key.startswith("_"):
            continue
        key_path = path + (key,)
        dotted = ".".join(key_path)
        if free_form:
            out[key] = value  # checked in _validate
            continue
        if key not in base:
            warnings.append(f"Unknown setting '{dotted}' ignored")
            continue
        default = base[key]
        if isinstance(default, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"'{dotted}' must be an object")
            out[key] = _merge(default, value, key_path, warnings)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"'{dotted}' must be true or false")
            out[key] = value
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"'{dotted}' must be a number")
            if isinstance(default, int) and not isinstance(value, int):
                raise ConfigError(f"'{dotted}' must be a whole number")
            out[key] = value
        else:
            if not isinstance(value, type(default)):
                raise ConfigError(f"'{dotted}' must be a {type(default).__name__}")
            out[key] = value
    return out


def _validate(cfg: dict) -> None:
    def get(path):
        node = cfg
        for p in path:
            node = node[p]
        return node

    for path, (lo, hi) in _RANGES.items():
        v = get(path)
        if not lo <= v <= hi:
            raise ConfigError(f"'{'.'.join(path)}' = {v} is outside {lo}..{hi}")
    for path, choices in _CHOICES.items():
        v = get(path)
        if v not in choices:
            raise ConfigError(f"'{'.'.join(path)}' = '{v}' must be one of {', '.join(choices)}")

    zone = cfg["active_zone"]
    if zone["left"] + zone["right"] >= 0.9 or zone["top"] + zone["bottom"] >= 0.9:
        raise ConfigError("active_zone margins leave too small an area")

    g = cfg["gestures"]
    if g["pinch_on_ratio"] >= g["pinch_off_ratio"]:
        raise ConfigError("gestures.pinch_on_ratio must be smaller than gestures.pinch_off_ratio")

    for section in ("hold_times_s", "cooldowns_s"):
        for name, v in cfg["timing"][section].items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ConfigError(f"'timing.{section}.{name}' must be a number >= 0")

    # Imported here to avoid importing pyautogui when only reading config.
    from src.actions import validate_action

    for name, spec in cfg["gesture_actions"].items():
        if spec is None:
            continue
        if not isinstance(spec, str):
            raise ConfigError(f"'gesture_actions.{name}' must be a string or null")
        err = validate_action(spec)
        if err:
            raise ConfigError(f"'gesture_actions.{name}': {err}")

    if cfg["gesture_actions"].get("two_hand_pinch") not in (None, "none", "zoom", "scroll"):
        raise ConfigError("'gesture_actions.two_hand_pinch' must be zoom, scroll or none")


def to_namespace(d: Any) -> Any:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    return d


def load_config(path: str | Path | None = None, overrides: dict | None = None):
    """Return (config_namespace, raw_dict, warnings)."""
    warnings: list[str] = []
    user: dict = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            try:
                user = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise ConfigError(f"{p.name} is not valid JSON (line {e.lineno}): {e.msg}") from e
            if not isinstance(user, dict):
                raise ConfigError(f"{p.name} must contain a JSON object")
        else:
            warnings.append(f"{p} not found, using built-in defaults")
    merged = _merge(DEFAULTS, user, (), warnings)
    if overrides:
        merged = _merge(merged, overrides, (), warnings)
    _validate(merged)
    ns = _namespace(merged)
    ns.profiles = {name: _build_profile(merged, name, prof, warnings) for name, prof in merged["profiles"].items()}
    return ns, merged, warnings


def _namespace(merged: dict):
    ns = to_namespace(merged)
    # Kept as dicts: these are looked up by gesture name.
    ns.gesture_actions = dict(merged["gesture_actions"])
    ns.timing.hold_times_s = dict(merged["timing"]["hold_times_s"])
    ns.timing.cooldowns_s = dict(merged["timing"]["cooldowns_s"])
    ns.profiles = {}
    return ns


def _build_profile(merged: dict, name: str, prof, warnings: list[str]):
    """Full config for one app profile: the base config with the profile's overrides."""
    where = f"profiles.{name}"
    if not isinstance(prof, dict):
        raise ConfigError(f"'{where}' must be an object")
    match = prof.get("match")
    if not isinstance(match, list) or not match or not all(isinstance(m, str) for m in match):
        raise ConfigError(f"'{where}.match' must be a list of program names, e.g. [\"chrome.exe\"]")
    overrides = {}
    for key, value in prof.items():
        if key == "match" or key.startswith("_"):
            continue
        if key not in PROFILE_SECTIONS:
            warnings.append(f"'{where}.{key}' can't be set per app and is ignored")
            continue
        overrides[key] = value
    full = _merge(merged, overrides, (), warnings)
    try:
        _validate(full)
    except ConfigError as e:
        raise ConfigError(f"in {where}: {e}") from e
    return SimpleNamespace(match=[m.lower() for m in match], cfg=_namespace(full))
