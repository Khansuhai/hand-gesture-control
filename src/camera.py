"""Webcam capture on a background thread.

Reading happens continuously in its own thread and only the newest frame is
kept, so slow processing never builds up a queue of old frames (= lag).
"""

from __future__ import annotations

import threading
import time

import cv2

_BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30, backend: str = "dshow"):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._read_seq = 0
        self._running = False
        self.failed = False
        self.size = (width, height)

    def _try_open(self, backend: str) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.index, _BACKENDS[backend])
        if not cap.isOpened():
            cap.release()
            return None
        if backend == "dshow":
            # MJPG lets most webcams deliver 30 FPS at this size over DirectShow.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(10):  # some cameras return a few empty frames at start
            ok, frame = cap.read()
            if ok and frame is not None:
                self.size = (frame.shape[1], frame.shape[0])
                return cap
            time.sleep(0.05)
        cap.release()
        return None

    def open(self) -> "Camera":
        order = [self.backend] + [b for b in ("dshow", "msmf", "any") if b != self.backend]
        for backend in order:
            cap = self._try_open(backend)
            if cap is not None:
                self._cap = cap
                self.backend = backend
                break
        else:
            raise CameraError(
                f"Could not open camera {self.index}. Close other apps that may be using it "
                "(Camera app, Zoom, Teams, browser tabs), check Windows Settings > Privacy & "
                "security > Camera, or try another index with --camera 1."
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        failures = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 50:
                    self.failed = True
                    with self._cond:
                        self._cond.notify_all()
                    return
                time.sleep(0.01)
                continue
            failures = 0
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 1.0):
        """Wait for a frame newer than the last one returned. None on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._read_seq or self.failed, timeout):
                return None
            if self.failed:
                return None
            self._read_seq = self._seq
            return self._frame

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()
