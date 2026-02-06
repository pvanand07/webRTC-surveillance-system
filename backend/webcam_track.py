"""
Webcam Stream Track - Custom aiortc VideoStreamTrack backed by WebcamCapture.
Converts numpy BGR frames to av.VideoFrame for WebRTC streaming.
"""

import numpy as np
from av import VideoFrame
from aiortc.mediastreams import VideoStreamTrack

from .webcam_capture import WebcamCapture


class WebcamStreamTrack(VideoStreamTrack):
    """Serves webcam frames to a WebRTC peer connection."""

    def __init__(self, capture: WebcamCapture):
        super().__init__()
        self._capture = capture

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        frame = self._capture.get_frame()
        if frame is None:
            frame = np.zeros((self._capture.height, self._capture.width, 3), dtype=np.uint8)

        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base
        return av_frame
