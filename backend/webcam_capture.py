"""
Webcam Capture - OpenCV-based webcam capture with background thread.
Distributes numpy BGR frames to async consumers.
"""

import asyncio
import platform
import threading
import time
import logging
from typing import Callable, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamCapture:
    """Captures webcam frames via OpenCV in a background thread and distributes to async consumers."""

    def __init__(
        self,
        device=None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        if device is None:
            system = platform.system()
            device = 0 if system == "Windows" else "/dev/video0"

        self.device = device
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._consumers: List[Callable] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.actual_fps: float = float(fps)  # from camera CAP_PROP_FPS (often wrong)
        self.measured_fps: float = float(fps)  # running average from capture loop

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Open the webcam and start the background capture thread."""
        self._loop = loop
        self._cap = cv2.VideoCapture(self.device)
        if isinstance(self.device, str) and self.device.startswith("/dev/video"):
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam: {self.device}")

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_fps = self._cap.get(cv2.CAP_PROP_FPS)

        # Use the camera-reported FPS if it looks valid, otherwise keep the
        # requested value.  Many USB cameras report 0 or very high values, so
        # we clamp to a sensible range.
        if 1 < reported_fps < 120:
            self.actual_fps = reported_fps
        else:
            self.actual_fps = float(self.fps)

        logger.info(
            f"Webcam opened: {self.device} ({actual_w}x{actual_h}, "
            f"reported={reported_fps:.1f}fps, using={self.actual_fps:.1f}fps)"
        )

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="webcam-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capture and release the webcam."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Webcam capture stopped")

    def add_consumer(self, callback: Callable) -> None:
        """Register an async consumer: async def cb(frame: np.ndarray, timestamp: float)"""
        self._consumers.append(callback)

    def get_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the latest captured frame (thread-safe)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running

    def _capture_loop(self) -> None:
        """Background thread: read frames from webcam and push to consumers."""
        frame_interval = 1.0 / self.fps
        frame_count = 0
        # #region agent log
        import json as _json
        _DBG_LOG = "/media/siva/MEDIA/DEV/OBJECT-DETECTION/.cursor/debug.log"
        _fps_window_start = time.monotonic()
        _fps_window_frames = 0
        # #endregion

        while self._running:
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            with self._lock:
                self._frame = frame

            frame_count += 1
            timestamp = time.monotonic()

            # #region agent log
            _fps_window_frames += 1
            _elapsed_window = timestamp - _fps_window_start
            if _elapsed_window >= 2.0:
                _measured_fps = _fps_window_frames / _elapsed_window
                # Exponential moving average so FPS is stable and reflects recent delivery
                self.measured_fps = 0.85 * self.measured_fps + 0.15 * _measured_fps
                try:
                    with open(_DBG_LOG, "a") as _f:
                        _f.write(_json.dumps({"hypothesisId":"H1","location":"webcam_capture.py:capture_loop","message":"measured_fps","data":{"measured_fps":round(_measured_fps,2),"frames_in_window":_fps_window_frames,"window_seconds":round(_elapsed_window,2),"total_frames":frame_count},"timestamp":int(time.time()*1000),"sessionId":"debug-session"}) + "\n")
                except Exception:
                    pass
                _fps_window_start = timestamp
                _fps_window_frames = 0
            # #endregion

            if self._loop is not None and self._consumers:
                for cb in self._consumers:
                    try:
                        asyncio.run_coroutine_threadsafe(cb(frame, timestamp), self._loop)
                    except Exception:
                        pass

            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"Capture thread exited after {frame_count} frames")
