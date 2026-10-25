"""Guided calibration: measures YOUR hand and camera, then saves thresholds.

Run with `python main.py --calibrate` (or calibrate.bat). It asks for four
poses (open hand, fist, point, pinch), records a couple of seconds of each,
and writes the finger / thumb / pinch thresholds into config.json. A backup
of the old file is kept as config.json.bak.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from src.gestures import (
    _LONG_FINGERS,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    THUMB_TIP,
    HandSelector,
    _d,
    _xy,
    extension_ratio,
    hand_size,
)

STEPS = [
    ("open", "Open hand: spread all 5 fingers"),
    ("fist", "Make a fist (thumb folded over fingers)"),
    ("point", "Point with index finger, others down"),
    ("pinch", "Pinch thumb + index together, hold"),
]
READY_S = 3.0
CAPTURE_S = 2.0
MIN_SAMPLES = 15


class CalibrationError(ValueError):
    pass


def measure(hand) -> dict:
    p = _xy(hand.landmarks)
    size = hand_size(p)
    return {
        "ext": {name: extension_ratio(p, pip, tip) for name, (pip, tip) in _LONG_FINGERS.items()},
        "thumb": _d(p, THUMB_TIP, INDEX_MCP) / size,
        "pinch": _d(p, THUMB_TIP, INDEX_TIP) / size,
        "index_ext": extension_ratio(p, INDEX_PIP, INDEX_TIP),
    }


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_thresholds(samples: dict[str, list[dict]]) -> tuple[dict, list[str]]:
    """samples: step name -> list of measure() dicts. Returns (gestures settings, notes)."""
    notes: list[str] = []
    for step, _ in STEPS:
        if len(samples.get(step, [])) < MIN_SAMPLES:
            raise CalibrationError(f"Not enough data for '{step}' - keep your hand in view.")
    med = lambda xs: float(np.median(xs))  # noqa: E731
    pct = lambda xs, q: float(np.percentile(xs, q))  # noqa: E731

    fingers = list(_LONG_FINGERS)
    open_ext = {f: med([s["ext"][f] for s in samples["open"]]) for f in fingers}
    fist_ext = {f: med([s["ext"][f] for s in samples["fist"]]) for f in fingers}
    hi, lo = min(open_ext.values()), max(fist_ext.values())
    if hi - lo < 0.1:
        raise CalibrationError("Open hand and fist look too similar - try better light or move closer.")
    out = {"finger_extended_ratio": round((hi + lo) / 2, 3)}

    thumb_open = med([s["thumb"] for s in samples["open"]])
    thumb_fist = med([s["thumb"] for s in samples["fist"]])
    if thumb_open - thumb_fist >= 0.1:
        out["thumb_extended_ratio"] = round((thumb_open + thumb_fist) / 2, 3)
    else:
        notes.append("Thumb open vs folded was unclear; kept the old thumb setting.")

    closed = pct([s["pinch"] for s in samples["pinch"]], 90)  # loosest pinch you made
    opened = pct([s["pinch"] for s in samples["point"]], 10)  # closest the thumb came while pointing
    if opened - closed < 0.1:
        raise CalibrationError("Pinch and point looked too similar - touch the fingertips firmly.")
    gap = opened - closed
    on = _clamp(closed + 0.35 * gap, 0.1, 0.45)
    off = _clamp(closed + 0.6 * gap, on + 0.04, 0.6)
    freeze = _clamp(closed + 0.8 * gap, off + 0.02, 0.8)
    out.update(pinch_on_ratio=round(on, 3), pinch_off_ratio=round(off, 3), pinch_freeze_ratio=round(freeze, 3))

    pinch_ext = med([s["index_ext"] for s in samples["pinch"]])
    min_ext = _clamp((pinch_ext + fist_ext["index"]) / 2, 0.7, 0.98)
    if min_ext < pinch_ext:
        out["pinch_min_extension"] = round(min_ext, 3)
    else:
        notes.append("Index finger was very bent while pinching; kept the old pinch extension setting.")
    return out, notes


def save_thresholds(config_path: Path, values: dict) -> Path | None:
    """Write values into config.json's "gestures" section. Returns the backup path."""
    data = {}
    backup = None
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        backup = config_path.with_name(config_path.name + ".bak")
        shutil.copyfile(config_path, backup)
    data.setdefault("gestures", {}).update(values)
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return backup


def run_calibration(cfg, config_path: Path, base_dir: Path) -> int:
    import cv2

    from src.camera import Camera, CameraError
    from src.hand_tracker import HandTracker
    from src.overlay import YELLOW, GREEN, RED, WHITE, _draw_hand, _text

    window = "Hand Gesture Calibration"
    camera = Camera(cfg.camera.index, cfg.camera.width, cfg.camera.height, cfg.camera.fps, cfg.camera.backend)
    tracker = HandTracker(cfg.tracking, base_dir)
    selector = HandSelector(cfg.tracking.hand, cfg.tracking.use_other_hand_if_missing, cfg.tracking.switch_delay_s)
    samples: dict[str, list[dict]] = {step: [] for step, _ in STEPS}
    try:
        camera.open()
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        print("Calibration: follow the instructions in the window. ESC cancels.")
        for step, instruction in STEPS:
            start = time.monotonic()
            while True:
                frame = camera.read(timeout=2.0)
                if frame is None:
                    if camera.failed:
                        raise CameraError("The camera stopped sending frames.")
                    continue
                now = time.monotonic()
                if cfg.camera.mirror:
                    frame = cv2.flip(frame, 1)
                hand = selector.select(tracker.process(frame, int(now * 1000)), now)
                elapsed = now - start
                if hand is not None:
                    _draw_hand(frame, hand, GREEN, main=True)
                if elapsed < READY_S:
                    status, color = f"Get ready... {READY_S - elapsed:.0f}", YELLOW
                elif elapsed < READY_S + CAPTURE_S:
                    status, color = "Hold still - measuring", GREEN
                    if hand is not None:
                        samples[step].append(measure(hand))
                elif len(samples[step]) < MIN_SAMPLES:
                    status, color = "Hand not seen - try again", RED
                    samples[step].clear()
                    start = now
                else:
                    break
                n = [s for s, _ in STEPS].index(step) + 1
                _text(frame, f"Step {n}/{len(STEPS)}: {instruction}", (10, 28), 0.58, WHITE, 2)
                _text(frame, status, (10, 58), 0.65, color, 2)
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("Calibration cancelled, nothing was changed.")
                    return 1

        try:
            values, notes = compute_thresholds(samples)
        except CalibrationError as e:
            print(f"Calibration failed: {e}\nNothing was changed. Please run it again.")
            return 1
        backup = save_thresholds(config_path, values)
        print("Calibration saved to", config_path)
        for k, v in values.items():
            print(f"  gestures.{k} = {v}")
        for note in notes:
            print("  Note:", note)
        if backup:
            print(f"Old settings backed up to {backup}")
        return 0
    except CameraError as e:
        print(f"Camera error: {e}")
        return 1
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()
