"""
Ring Buffer - Circular buffer for storing recent video frames.
Maintains last N seconds of frames for pre-roll recording.
"""

from collections import deque
from typing import List, Optional
from datetime import datetime
import asyncio
import numpy as np


class FrameData:
    """Container for a single buffered frame."""

    __slots__ = ("timestamp", "frame_array")

    def __init__(self, timestamp: datetime, frame_array: np.ndarray):
        self.timestamp = timestamp
        self.frame_array = frame_array


class RingBuffer:
    """Circular buffer that keeps the last *duration_seconds* of video frames."""

    def __init__(self, duration_seconds: float = 5.0, fps: float = 30.0):
        self.duration_seconds = duration_seconds
        self.fps = fps
        self.max_frames = int(duration_seconds * fps)
        self.buffer: deque[FrameData] = deque(maxlen=self.max_frames)
        self.lock = asyncio.Lock()

    async def add_frame(self, frame_array: np.ndarray) -> None:
        """Append a BGR numpy frame to the buffer."""
        async with self.lock:
            self.buffer.append(FrameData(
                timestamp=datetime.utcnow(),
                frame_array=frame_array,
            ))

    async def get_preroll_frames(self, seconds: float = None) -> List[np.ndarray]:
        """Return numpy arrays from the last *seconds* of buffered frames."""
        async with self.lock:
            if not self.buffer:
                return []
            if seconds is None:
                seconds = self.duration_seconds
            num_frames = min(int(seconds * self.fps), len(self.buffer))
            return [f.frame_array for f in list(self.buffer)[-num_frames:]]

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
