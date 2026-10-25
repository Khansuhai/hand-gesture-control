"""Draws the preview: hand skeleton, active zone, gesture, mode, FPS."""

from __future__ import annotations

import cv2
import numpy as np

from src.gestures import FINGER_NAMES, INDEX_TIP, THUMB_TIP, GestureResult, Hand

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

GREEN = (80, 220, 80)
RED = (60, 60, 230)
YELLOW = (0, 210, 255)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)
CYAN = (230, 200, 40)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, text, org, scale=0.55, color=WHITE, thick=1):
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _draw_hand(img, hand: Hand, color, main: bool):
    pts = hand.landmarks[:, :2].astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2 if main else 1, cv2.LINE_AA)
    for i, p in enumerate(pts):
        r = 6 if main and i in (THUMB_TIP, INDEX_TIP) else 3
        cv2.circle(img, tuple(p), r, WHITE if main else GREY, -1, cv2.LINE_AA)


_FINGER_TIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky


def _draw_finger_names(img, hand: Hand, result: GestureResult):
    pts = hand.landmarks[:, :2].astype(int)
    for name, tip in zip(FINGER_NAMES, _FINGER_TIPS):
        up = result.features.fingers[name]
        x, y = pts[tip]
        _text(img, name.capitalize(), (x - 18, y - 12), 0.42, GREEN if up else GREY, 1)


def draw(
    img: np.ndarray,
    *,
    zone,
    all_hands: list[Hand],
    hand: Hand | None,
    result: GestureResult | None,
    controller,
    fps: float,
    now: float,
    dry_run: bool,
    show_debug: bool,
    show_help: bool = False,
) -> np.ndarray:
    h, w = img.shape[:2]

    # Active zone: this rectangle maps to the whole screen.
    x1, y1 = int(zone.left * w), int(zone.top * h)
    x2, y2 = int((1 - zone.right) * w), int((1 - zone.bottom) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, 1, cv2.LINE_AA)

    for other in all_hands:
        if other is not hand:
            _draw_hand(img, other, GREY, main=False)
    if hand is not None:
        _draw_hand(img, hand, CYAN if not controller.paused else GREY, main=True)
        if result is not None:
            _draw_finger_names(img, hand, result)
        if result is not None and controller.dragging:
            px, py = result.pinch_point
            cv2.circle(img, (int(px * w), int(py * h)), 10, RED, 2, cv2.LINE_AA)

    # Top bar.
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.45, img, 0.55, 0, img)

    mode = "DRY RUN" if dry_run and not controller.paused else controller.mode
    mode_color = RED if controller.paused else (YELLOW if dry_run else GREEN)
    _text(img, mode, (10, 24), 0.7, mode_color, 2)
    pose = result.pose if result is not None else "no hand"
    if controller.two_hand_active:
        pose = "two-hand zoom"
    elif controller.dragging:
        pose = "pinch (DRAGGING)"
    elif controller.rotating:
        pose = f"{pose} (ROTATING)"
    _text(img, f"Gesture: {pose}", (130, 24), 0.6)
    if result is not None:
        _text(img, f"Fingers up: {result.features.count()}", (w - 165, h - 30), 0.6, GREEN)
    _text(img, f"FPS {fps:4.1f}", (w - 105, 24), 0.6, GREEN if fps >= 25 else YELLOW)

    text, t = controller.last_action
    if text and now - t < 1.2:
        _text(img, text, (10, 50), 0.65, YELLOW, 2)

    hp = controller.hold_progress
    if hp is not None:
        name, frac = hp
        bx, by, bw = w - 210, 40, 200
        cv2.rectangle(img, (bx, by), (bx + bw, by + 10), GREY, 1)
        cv2.rectangle(img, (bx, by), (bx + int(bw * frac), by + 10), YELLOW, -1)
        _text(img, name, (bx - 95, by + 10), 0.45)

    if show_debug and result is not None:
        f = result.features
        fingers = " ".join(n[0].upper() if f.fingers[n] else "." for n in FINGER_NAMES)
        lines = [
            f"fingers [{fingers}]  thumb_up={'Y' if f.thumb_up else 'n'}",
            f"pinch {f.pinch_ratio:.2f}  mid-pinch {f.middle_pinch_ratio:.2f}  raw={result.raw_pose}",
            f"hand {hand.handedness if hand else '?'}  size {f.hand_size:.0f}px",
        ]
        for i, line in enumerate(lines):
            _text(img, line, (10, h - 66 + i * 18), 0.45, WHITE)
    elif hand is None and all_hands:
        _text(img, "Hand detected but not in control (see 'tracking' in config.json)", (10, h - 34), 0.45, YELLOW)

    if controller.profile != "default":
        _text(img, f"App profile: {controller.profile}", (w - 210, 76), 0.45, CYAN)

    if show_help:
        _draw_help(img, controller.actions, controller.profile)
    _text(img, "H help   Ctrl+Alt+Q / ESC quit   Ctrl+Alt+P pause", (w - 395, h - 8), 0.42, GREY)
    return img


# Gesture name -> how to do it, in the order shown in the help panel.
HOW_TO = [
    ("point", "Index finger up"),
    ("pinch", "Thumb + index tap"),
    ("double_pinch", "Two quick taps"),
    ("pinch_hold", "Thumb + index hold"),
    ("middle_pinch", "Thumb + middle tap"),
    ("two_fingers", "V sign, move up/down"),
    ("three_fingers", "3 fingers, move up/down"),
    ("four_fingers", "4 fingers, move"),
    ("two_hand_pinch", "Both hands pinch, pull/push"),
    ("call_me", "Thumb + pinky, move up/down"),
    ("thumbs_up", "Thumbs up (hold)"),
    ("thumbs_down", "Thumbs down (hold)"),
    ("rock", "Index + pinky (hold)"),
    ("pinky", "Pinky only (hold)"),
    ("swipe_left", "Open palm swipe left"),
    ("swipe_right", "Open palm swipe right"),
    ("swipe_up", "Open palm swipe up"),
    ("swipe_down", "Open palm swipe down"),
    ("fist", "Fist (hold)"),
    ("open_palm", "Open palm (hold)"),
]


def _draw_help(img, actions: dict, profile: str):
    h, w = img.shape[:2]
    rows = [(how, actions.get(name)) for name, how in HOW_TO]
    rows = [(how, a) for how, a in rows if a not in (None, "none", "")]
    rows += [(name, a) for name, a in actions.items() if name not in dict(HOW_TO) and a not in (None, "none", "")]
    panel = img.copy()
    cv2.rectangle(panel, (20, 66), (w - 20, h - 24), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.78, img, 0.22, 0, img)
    _text(img, f"GESTURES  (profile: {profile})   press H to close", (32, 88), 0.5, YELLOW)
    y = 110
    for how, action in rows:
        if y > h - 30:
            break
        _text(img, how, (32, y), 0.43, WHITE)
        _text(img, str(action), (330, y), 0.43, GREEN)
        y += 17
