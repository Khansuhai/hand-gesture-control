"""Turn 21 hand landmarks into a named pose ("point", "pinch", "fist", ...).

Everything here is pure math on numpy arrays, so it can be unit-tested with
synthetic hands and never needs a camera.

Landmark indices (MediaPipe):
    0 wrist
    1-4  thumb  (CMC, MCP, IP, TIP)
    5-8  index  (MCP, PIP, DIP, TIP)
    9-12 middle, 13-16 ring, 17-20 pinky (same order)

All distances are divided by the hand size, so thresholds work the same when
the hand is near or far from the camera.

Adding a new pose: write a function `(HandFeatures) -> bool` and call
`register_pose("my_pose", fn)` (or add it to POSES). Then map it to an action
in config.json, e.g. "my_pose": "key:volumeup", and give it a hold time in
timing.hold_times_s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
# (PIP, TIP) for the four long fingers.
_LONG_FINGERS = {
    "index": (INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_PIP, RING_TIP),
    "pinky": (PINKY_PIP, PINKY_TIP),
}

# Typical palm width (index MCP to pinky MCP) relative to wrist->middle MCP.
_PALM_WIDTH_TO_SIZE = 0.75


@dataclass
class Hand:
    """One detected hand.

    landmarks: (21, 3) array in pixels (x right, y down; z roughly in x units).
    image_size: (width, height) of the frame the landmarks belong to.
    """

    landmarks: np.ndarray
    handedness: str = "Right"
    score: float = 1.0
    image_size: tuple[int, int] = (640, 480)

    def norm(self, idx: int) -> tuple[float, float]:
        """Landmark position as a 0..1 fraction of the frame."""
        w, h = self.image_size
        return float(self.landmarks[idx, 0] / w), float(self.landmarks[idx, 1] / h)

    def norm_mid(self, a: int, b: int) -> tuple[float, float]:
        ax, ay = self.norm(a)
        bx, by = self.norm(b)
        return (ax + bx) / 2, (ay + by) / 2


@dataclass
class HandFeatures:
    fingers: dict[str, bool]
    thumb_up: bool
    hand_size: float
    pinch_ratio: float
    middle_pinch_ratio: float
    index_pinched: bool
    middle_pinched: bool
    # Fingers closing in on a pinch (used to freeze the cursor just before a click).
    index_pinch_near: bool = False
    middle_pinch_near: bool = False
    thumb_down: bool = False

    def four_curled(self) -> bool:
        return not any(self.fingers[f] for f in ("index", "middle", "ring", "pinky"))

    def count(self) -> int:
        """How many fingers are up (0-5)."""
        return sum(self.fingers.values())


@dataclass
class GestureResult:
    pose: str  # stable pose (what the controller acts on)
    raw_pose: str  # pose in this frame only
    features: HandFeatures
    index_tip: tuple[float, float]  # normalized 0..1
    pinch_point: tuple[float, float]  # between thumb tip and index tip
    palm_point: tuple[float, float]  # middle-finger knuckle, very stable
    hand: Hand = field(repr=False, default=None)


# ---- pose rules ------------------------------------------------------------
# Checked top to bottom; the first rule that matches wins.

PoseRule = Callable[[HandFeatures], bool]

POSES: list[tuple[str, PoseRule]] = [
    ("pinch", lambda f: f.index_pinched),
    ("middle_pinch", lambda f: f.middle_pinched),
    ("open_palm", lambda f: all(f.fingers.values())),
    (
        "four_fingers",
        lambda f: all(f.fingers[n] for n in ("index", "middle", "ring", "pinky")) and not f.fingers["thumb"],
    ),
    (
        "three_fingers",
        lambda f: f.fingers["index"] and f.fingers["middle"] and f.fingers["ring"] and not f.fingers["pinky"],
    ),
    ("thumbs_up", lambda f: f.four_curled() and f.thumb_up),
    ("thumbs_down", lambda f: f.four_curled() and f.thumb_down),
    ("fist", lambda f: f.four_curled() and not f.thumb_up and not f.thumb_down),
    (
        # 🤙 thumb and pinky out
        "call_me",
        lambda f: f.fingers["thumb"] and f.fingers["pinky"]
        and not f.fingers["index"] and not f.fingers["middle"] and not f.fingers["ring"],
    ),
    (
        # 🤘 index and pinky up, thumb tucked
        "rock",
        lambda f: f.fingers["index"] and f.fingers["pinky"]
        and not f.fingers["middle"] and not f.fingers["ring"] and not f.fingers["thumb"],
    ),
    (
        "pinky",
        lambda f: f.fingers["pinky"] and not f.fingers["thumb"]
        and not f.fingers["index"] and not f.fingers["middle"] and not f.fingers["ring"],
    ),
    (
        "two_fingers",
        lambda f: f.fingers["index"] and f.fingers["middle"]
        and not f.fingers["ring"] and not f.fingers["pinky"],
    ),
    (
        "point",
        lambda f: f.fingers["index"]
        and not f.fingers["middle"] and not f.fingers["ring"] and not f.fingers["pinky"],
    ),
]


def register_pose(name: str, rule: PoseRule, before: str | None = None) -> None:
    """Add a custom pose. `before` puts it ahead of an existing pose in priority."""
    if before is None:
        POSES.append((name, rule))
        return
    for i, (existing, _) in enumerate(POSES):
        if existing == before:
            POSES.insert(i, (name, rule))
            return
    raise KeyError(before)


# ---- geometry --------------------------------------------------------------


def _xy(lm: np.ndarray) -> np.ndarray:
    return lm[:, :2].astype(float)


def _d(p: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(p[a] - p[b]))


def hand_size(p: np.ndarray) -> float:
    """Max of wrist->middle knuckle and scaled palm width.

    Using both keeps the size stable when the hand tilts toward the camera
    (wrist->knuckle shrinks) or turns sideways (palm width shrinks).
    """
    return max(_d(p, WRIST, MIDDLE_MCP), _d(p, INDEX_MCP, PINKY_MCP) / _PALM_WIDTH_TO_SIZE, 1e-6)


def extension_ratio(p: np.ndarray, pip: int, tip: int) -> float:
    """>1 when the fingertip is further from the wrist than the middle joint."""
    return _d(p, WRIST, tip) / max(_d(p, WRIST, pip), 1e-6)


class Stabilizer:
    """A value must repeat for `frames` frames in a row before it is accepted."""

    def __init__(self, frames: int, initial: str = "none"):
        self.frames = max(1, frames)
        self.initial = initial
        self.reset()

    def reset(self) -> None:
        self.stable = self.initial
        self._candidate = self.initial
        self._count = 0

    def update(self, value: str) -> str:
        if value == self._candidate:
            self._count += 1
        else:
            self._candidate = value
            self._count = 1
        if self._count >= self.frames:
            self.stable = value
        return self.stable


class HandSelector:
    """Pick which detected hand is in control.

    The preferred hand always wins when visible. If it is gone and
    `allow_other` is on, the other hand takes over - but only after the
    preferred hand has been missing for `switch_delay_s`, so a one-frame
    dropout (or MediaPipe briefly mislabeling a hand) doesn't flip control.
    """

    # A lone hand within this distance (fraction of the frame) of where the
    # controlling hand just was is the same hand, whatever its label says.
    SAME_HAND_DIST = 0.15
    SAME_HAND_MEMORY_S = 0.5

    def __init__(self, preferred: str = "right", allow_other: bool = True, switch_delay_s: float = 0.3):
        self.preferred = preferred  # right | left | any
        self.allow_other = allow_other
        self.switch_delay_s = switch_delay_s
        self.active: str | None = None
        self.switched = False  # True on the frame control moves to a different physical hand
        self._missing_since: float | None = None
        self._last_pos: np.ndarray | None = None
        self._last_label: str | None = None
        self._last_time = -1e9

    def select(self, hands: list[Hand], now: float) -> Hand | None:
        self.switched = False
        chosen = self._choose(hands, now)
        if chosen is not None:
            pos = np.array(chosen.norm(WRIST))
            label = chosen.handedness.lower()
            if self._last_label is not None and label != self._last_label:
                # A new label is only a new hand if it's somewhere else (or
                # both hands are in view). Otherwise MediaPipe just relabeled
                # the same hand, which happens when it turns.
                far = self._last_pos is None or np.linalg.norm(pos - self._last_pos) > self.SAME_HAND_DIST
                self.switched = far or len(hands) > 1
            self._last_pos, self._last_label, self._last_time = pos, label, now
        return chosen

    def _choose(self, hands: list[Hand], now: float) -> Hand | None:
        if (
            len(hands) == 1
            and self._last_pos is not None
            and now - self._last_time <= self.SAME_HAND_MEMORY_S
            and np.linalg.norm(np.array(hands[0].norm(WRIST)) - self._last_pos) <= self.SAME_HAND_DIST
        ):
            return hands[0]  # same hand as before, even if relabeled

        by_label: dict[str, Hand] = {}
        for h in sorted(hands, key=lambda h: h.score):  # best score per label wins
            by_label[h.handedness.lower()] = h

        if self.preferred == "any":
            wanted = self.active
        else:
            wanted = self.preferred
        if wanted in by_label:
            self._missing_since = None
            self.active = wanted
            return by_label[wanted]

        if self._missing_since is None:
            self._missing_since = now
        if self.preferred != "any" and not self.allow_other:
            return None
        if not by_label:
            return None
        if self.active is not None and self.active in by_label:
            return by_label[self.active]  # already switched over
        if self.active is None or now - self._missing_since >= self.switch_delay_s:
            best = max(by_label.values(), key=lambda h: h.score)
            self.active = best.handedness.lower()
            return best
        return None


class GestureRecognizer:
    def __init__(self, gcfg, lost_reset_s: float = 0.3):
        self.cfg = gcfg
        self.lost_reset_s = lost_reset_s
        self.stabilizer = Stabilizer(gcfg.stable_frames)
        self._last_seen: float | None = None
        self.reset()

    def reset(self) -> None:
        self._index_pinched = False
        self._middle_pinched = False
        self.stabilizer.reset()

    def features(self, hand: Hand) -> HandFeatures:
        c = self.cfg
        p = _xy(hand.landmarks)
        size = hand_size(p)

        fingers = {
            name: extension_ratio(p, pip, tip) > c.finger_extended_ratio
            for name, (pip, tip) in _LONG_FINGERS.items()
        }
        thumb_out = (
            _d(p, THUMB_TIP, INDEX_MCP) / size > c.thumb_extended_ratio
            and _d(p, THUMB_TIP, PINKY_MCP) > _d(p, THUMB_IP, PINKY_MCP)
        )
        fingers = {"thumb": thumb_out, **fingers}

        thumb_vec = p[THUMB_TIP] - p[THUMB_MCP]
        alignment = -thumb_vec[1] / max(float(np.linalg.norm(thumb_vec)), 1e-6)
        thumb_highest = p[THUMB_TIP, 1] <= p[INDEX_MCP:, 1].min()
        thumb_up = thumb_out and alignment >= c.thumb_up_min_alignment and thumb_highest
        thumb_lowest = p[THUMB_TIP, 1] >= p[INDEX_MCP:, 1].max()
        thumb_down = thumb_out and -alignment >= c.thumb_up_min_alignment and thumb_lowest

        pinch_ratio = _d(p, THUMB_TIP, INDEX_TIP) / size
        middle_ratio = _d(p, THUMB_TIP, MIDDLE_TIP) / size

        # Hysteresis: easy to stay pinched, a bit harder to start. The
        # extension check stops a fist (thumb resting on curled index) from
        # looking like a pinch.
        def pinched(was: bool, ratio: float, pip: int, tip: int) -> bool:
            if was:
                return ratio < c.pinch_off_ratio
            return ratio < c.pinch_on_ratio and extension_ratio(p, pip, tip) > c.pinch_min_extension

        ip = pinched(self._index_pinched, pinch_ratio, INDEX_PIP, INDEX_TIP)
        mp = pinched(self._middle_pinched, middle_ratio, MIDDLE_PIP, MIDDLE_TIP)
        if ip and mp:
            if self._index_pinched:
                mp = False
            elif self._middle_pinched:
                ip = False
            elif pinch_ratio <= middle_ratio:
                mp = False
            else:
                ip = False
        self._index_pinched, self._middle_pinched = ip, mp

        # Only counts if the finger is reaching out to the thumb. When pointing,
        # the thumb often rests on the curled middle finger - that is not a
        # right click coming and must not freeze the cursor.
        def near(ratio: float, pip: int, tip: int) -> bool:
            return ratio < c.pinch_freeze_ratio and extension_ratio(p, pip, tip) > c.pinch_min_extension

        return HandFeatures(
            fingers=fingers,
            thumb_up=bool(thumb_up),
            hand_size=size,
            pinch_ratio=pinch_ratio,
            middle_pinch_ratio=middle_ratio,
            index_pinched=ip,
            middle_pinched=mp,
            index_pinch_near=near(pinch_ratio, INDEX_PIP, INDEX_TIP),
            middle_pinch_near=near(middle_ratio, MIDDLE_PIP, MIDDLE_TIP),
            thumb_down=bool(thumb_down),
        )

    @staticmethod
    def classify(f: HandFeatures) -> str:
        for name, rule in POSES:
            if rule(f):
                return name
        return "none"

    def update(self, hand: Hand | None, now: float | None = None) -> GestureResult | None:
        if hand is None:
            # Short dropouts keep state (so a drag survives one bad frame);
            # longer ones start fresh.
            if self._last_seen is None or now is None or now - self._last_seen > self.lost_reset_s:
                self.reset()
            return None
        self._last_seen = now
        f = self.features(hand)
        raw = self.classify(f)
        stable = self.stabilizer.update(raw)
        return GestureResult(
            pose=stable,
            raw_pose=raw,
            features=f,
            index_tip=hand.norm(INDEX_TIP),
            pinch_point=hand.norm_mid(THUMB_TIP, INDEX_TIP),
            palm_point=hand.norm(MIDDLE_MCP),
            hand=hand,
        )


class SwipeDetector:
    """Quick open-palm swipe left/right/up/down.

    Watches the palm position over the last `max_time_s`. A swipe is a move
    of at least `min_distance` (fraction of the frame) in that time, mostly in
    one direction. A still palm (e.g. "resume") never triggers it.
    """

    POSE = "open_palm"

    def __init__(self, scfg):
        self.cfg = scfg
        self._history: list[tuple[float, float, float]] = []
        self._misses = 0
        self._blocked_until = -1e9

    def reset(self) -> None:
        self._history.clear()
        self._misses = 0

    def update(self, result: GestureResult | None, now: float) -> str | None:
        c = self.cfg
        if result is None or result.raw_pose != self.POSE:
            # Fast motion blurs the image; allow a couple of odd frames.
            self._misses += 1
            if self._misses > 2:
                self.reset()
            return None
        self._misses = 0
        x, y = result.palm_point
        self._history.append((now, x, y))
        self._history = [s for s in self._history if now - s[0] <= c.max_time_s]
        if now < self._blocked_until or len(self._history) < 3:
            return None
        for t0, x0, y0 in self._history:
            dx, dy = x - x0, y - y0
            if abs(dx) >= c.min_distance and abs(dx) > 2 * abs(dy):
                direction = "swipe_right" if dx > 0 else "swipe_left"
            elif abs(dy) >= c.min_distance and abs(dy) > 2 * abs(dx):
                direction = "swipe_down" if dy > 0 else "swipe_up"
            else:
                continue
            self.reset()
            self._blocked_until = now + c.cooldown_s
            return direction
        return None


class TwoHandPinch:
    """Both hands pinching (thumb + index) at the same time.

    Returns the distance between the two pinch points (fraction of frame
    width) while both are pinched for `stable_frames` frames, else None.
    Hands are told apart by position (left/right in the frame), not by
    MediaPipe's handedness label, so mislabeling can't break it.
    """

    def __init__(self, gcfg):
        self.cfg = gcfg
        self._pinched = [False, False]
        self._count = 0

    def reset(self) -> None:
        self._pinched = [False, False]
        self._count = 0

    def update(self, hands: list[Hand]) -> float | None:
        if len(hands) < 2:
            self.reset()
            return None
        two = sorted(sorted(hands, key=lambda h: h.score)[-2:], key=lambda h: h.landmarks[WRIST, 0])
        points = []
        for i, hand in enumerate(two):
            p = _xy(hand.landmarks)
            ratio = _d(p, THUMB_TIP, INDEX_TIP) / hand_size(p)
            limit = self.cfg.pinch_off_ratio if self._pinched[i] else self.cfg.pinch_on_ratio
            self._pinched[i] = ratio < limit
            points.append((p[THUMB_TIP] + p[INDEX_TIP]) / 2)
        if not all(self._pinched):
            self._count = 0
            return None
        self._count += 1
        if self._count < self.cfg.stable_frames:
            return None
        width = two[0].image_size[0]
        return float(np.linalg.norm(points[0] - points[1]) / width)
