"""Build fake (synthetic) hands for tests - no camera needed.

Coordinates are in "hand units": wrist at (0, 0), wrist -> middle knuckle
is 1.0 long, fingers point up (negative y, like image coordinates).
"""

from __future__ import annotations

import math

import numpy as np

from src.gestures import Hand

THUMB = {
    "out": [(-0.3, -0.2), (-0.55, -0.4), (-0.85, -0.5), (-1.1, -0.6)],
    "tucked": [(-0.3, -0.2), (-0.45, -0.45), (-0.3, -0.7), (-0.08, -0.8)],
    "pinch_index": [(-0.3, -0.2), (-0.55, -0.45), (-0.68, -0.85), (-0.6, -1.26)],
    "pinch_middle": [(-0.3, -0.2), (-0.55, -0.45), (-0.6, -0.95), (-0.36, -1.33)],
}

FINGERS = {
    "index": {
        "ext": [(-0.3, -0.95), (-0.35, -1.35), (-0.38, -1.6), (-0.4, -1.82)],
        "curl": [(-0.3, -0.95), (-0.32, -1.2), (-0.3, -1.0), (-0.28, -0.8)],
        "pinch": [(-0.3, -0.95), (-0.35, -1.35), (-0.48, -1.42), (-0.58, -1.32)],
    },
    "middle": {
        "ext": [(0.0, -1.0), (0.0, -1.45), (0.0, -1.72), (0.0, -1.97)],
        "curl": [(0.0, -1.0), (0.0, -1.25), (0.0, -1.05), (0.01, -0.85)],
        "pinch": [(0.0, -1.0), (-0.05, -1.42), (-0.2, -1.48), (-0.33, -1.38)],
    },
    "ring": {
        "ext": [(0.25, -0.95), (0.28, -1.35), (0.3, -1.6), (0.32, -1.8)],
        "curl": [(0.25, -0.95), (0.27, -1.18), (0.26, -1.0), (0.25, -0.82)],
    },
    "pinky": {
        "ext": [(0.45, -0.85), (0.52, -1.15), (0.56, -1.33), (0.6, -1.5)],
        "curl": [(0.45, -0.85), (0.48, -1.05), (0.47, -0.9), (0.45, -0.75)],
    },
}


def make_hand(
    thumb: str = "tucked",
    index: str = "ext",
    middle: str = "curl",
    ring: str = "curl",
    pinky: str = "curl",
    *,
    scale: float = 100.0,
    center: tuple[float, float] = (320.0, 400.0),
    rotation_deg: float = 0.0,
    thumb_tip: tuple[float, float] | None = None,
    image_size: tuple[int, int] = (640, 480),
    handedness: str = "Right",
) -> Hand:
    """rotation_deg rotates clockwise on screen. `center` is where the wrist goes (pixels)."""
    pts = [(0.0, 0.0)]
    pts += THUMB[thumb]
    pts += FINGERS["index"][index]
    pts += FINGERS["middle"][middle]
    pts += FINGERS["ring"][ring]
    pts += FINGERS["pinky"][pinky]
    p = np.array(pts, dtype=float)
    if thumb_tip is not None:
        p[4] = thumb_tip
    a = math.radians(rotation_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = p @ rot.T * scale + np.array(center)
    lm = np.zeros((21, 3))
    lm[:, :2] = p
    return Hand(landmarks=lm, handedness=handedness, score=0.99, image_size=image_size)


# Ready-made poses (keyword arguments for make_hand).
POSES = {
    "point": dict(thumb="tucked", index="ext"),
    "two_fingers": dict(thumb="tucked", index="ext", middle="ext"),
    "open_palm": dict(thumb="out", index="ext", middle="ext", ring="ext", pinky="ext"),
    "four_fingers": dict(thumb="tucked", index="ext", middle="ext", ring="ext", pinky="ext"),
    "three_fingers": dict(thumb="tucked", index="ext", middle="ext", ring="ext"),
    "fist": dict(thumb="tucked", index="curl"),
    "thumbs_up": dict(thumb="out", index="curl", rotation_deg=90),
    "thumbs_down": dict(thumb="out", index="curl", rotation_deg=-90),
    "call_me": dict(thumb="out", index="curl", pinky="ext"),
    "rock": dict(thumb="tucked", index="ext", pinky="ext"),
    "pinky": dict(thumb="tucked", index="curl", pinky="ext"),
    "pinch": dict(thumb="pinch_index", index="pinch"),
    "middle_pinch": dict(thumb="pinch_middle", index="ext", middle="pinch"),
}


def pose_hand(name: str, **overrides) -> Hand:
    kwargs = dict(POSES[name])
    kwargs.update(overrides)
    return make_hand(**kwargs)
