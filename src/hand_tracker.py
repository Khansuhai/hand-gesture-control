"""MediaPipe HandLandmarker (Tasks API) wrapper.

Downloads the model on first run, runs in VIDEO mode (uses the previous
frame to track faster) and returns the detected hands.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src.gestures import Hand

_MIN_MODEL_BYTES = 1_000_000


def ensure_model(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size >= _MIN_MODEL_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Downloading hand model (~8 MB, one time only) from\n  {url}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 16):
                out.write(chunk)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the hand model ({e}). Check your internet connection, or "
            f"download it manually from the URL above and save it as {path}"
        ) from e
    if tmp.stat().st_size < _MIN_MODEL_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded hand model looks incomplete, please try again.")
    tmp.replace(path)
    print(f"Saved model to {path}")
    return path


class HandTracker:
    def __init__(self, tcfg, base_dir: Path):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        model_path = Path(tcfg.model_path)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
        ensure_model(model_path, tcfg.model_url)

        options = vision.HandLandmarkerOptions(
            # Passing bytes instead of a path avoids MediaPipe problems with
            # spaces/non-ASCII characters in Windows user folder names.
            base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=tcfg.max_hands,
            min_hand_detection_confidence=tcfg.min_detection_confidence,
            min_hand_presence_confidence=tcfg.min_presence_confidence,
            min_tracking_confidence=tcfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        """Return every hand in the frame. Which one is in control is decided
        by gestures.HandSelector.

        Frame must already be mirrored (MediaPipe's handedness assumes a selfie view).
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(timestamp_ms), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)

        hands: list[Hand] = []
        for i, lms in enumerate(result.hand_landmarks):
            cat = result.handedness[i][0] if i < len(result.handedness) and result.handedness[i] else None
            arr = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=float)
            hands.append(
                Hand(
                    landmarks=arr,
                    handedness=(cat.category_name if cat else "Unknown") or "Unknown",
                    score=float(cat.score) if cat and cat.score is not None else 0.0,
                    image_size=(w, h),
                )
            )
        return hands

    def close(self) -> None:
        self._landmarker.close()
