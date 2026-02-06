"""
Webcam Stream Track - Custom aiortc VideoStreamTrack backed by WebcamCapture.
Converts numpy BGR frames to av.VideoFrame for WebRTC streaming.
Uses annotated frame (e.g. YOLO boxes) when provided.
"""

import numpy as np
from av import VideoFrame
from aiortc.mediastreams import VideoStreamTrack
from typing import Optional

from .webcam_capture import WebcamCapture


class WebcamStreamTrack(VideoStreamTrack):
    """Serves webcam frames to a WebRTC peer connection. Uses annotated frame if available."""

    def __init__(self, capture: WebcamCapture, display_frame_holder: Optional[dict] = None):
        super().__init__()
        self._capture = capture
        self._display_frame_holder = display_frame_holder  # {"frame": np.ndarray | None}

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        frame = None
        if self._display_frame_holder and self._display_frame_holder.get("frame") is not None:
            frame = self._display_frame_holder["frame"]

        if frame is None:
            frame = self._capture.get_frame()

        if frame is None:
            frame = np.zeros((self._capture.height, self._capture.width, 3), dtype=np.uint8)

        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base
        return av_frame
