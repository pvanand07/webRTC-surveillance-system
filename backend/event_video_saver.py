"""
Event Video Saver - Saves video when YOLO detections occur.
Uses existing VideoRecorder and RingBuffer. One active recording at a time.
Start: lock onto a single tracking_id when first seen.
Stop: that exact tracking_id has been absent for 3.5s.
Saves metadata as JSON next to each clip.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Set

if TYPE_CHECKING:
    from .ring_buffer import RingBuffer
    from .video_recorder import VideoRecorder

logger = logging.getLogger(__name__)

# Seconds to keep recording after the tracked object (single track_id) disappears
RECORDING_BUFFER_SECONDS = 3.5


class EventVideoSaver:
    """
    Event-based recording driven by a single tracking_id:
    - Start when we see at least one tracked object; lock onto one ID for this event.
    - Stop when that exact tracking_id has been absent for recording_buffer_seconds.
    Only one active recording at a time.
    """

    def __init__(
        self,
        video_recorder: "VideoRecorder",
        ring_buffer: "RingBuffer",
        recording_buffer_seconds: float = RECORDING_BUFFER_SECONDS,
        preroll_seconds: float = 5.0,
        on_recording_saved: Optional[Callable[[dict], Awaitable[None]]] = None,
        on_recording_started: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.video_recorder = video_recorder
        self.ring_buffer = ring_buffer
        self.recording_buffer_seconds = recording_buffer_seconds
        self.preroll_seconds = preroll_seconds
        self._on_recording_saved = on_recording_saved
        self._on_recording_started = on_recording_started

        self._event_track_id: Optional[int] = None  # single ID we follow for this event
        self._last_detection_time: Optional[datetime] = None  # last time we saw _event_track_id
        self._detected_classes: Set[str] = set()
        self._class_detection_counts: dict = {}
        self._event_start_time: Optional[datetime] = None
        self._frame_metadata: list = []
        self._frame_count_at_stop: int = 0
        self._event_clip_id: Optional[str] = None

    @property
    def is_recording(self) -> bool:
        return (
            self.video_recorder.current_recording is not None
            and self.video_recorder.current_recording.get("event_recording") is True
        )

    @property
    def current_clip_id(self) -> Optional[str]:
        return self._event_clip_id if self.is_recording else None

    def _clip_id_for_event(self, detected_classes: Set[str]) -> str:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        classes_str = "_".join(sorted(detected_classes)) if detected_classes else "detection"
        return f"{classes_str}_{timestamp}"

    async def start_event_recording(
        self,
        event_track_id: int,
        detected_classes: Set[str],
        fps: float,
        frame_info: Optional[dict] = None,
    ) -> None:
        """Start a new event recording for the given tracking_id. Only one active at a time."""
        if self.video_recorder.current_recording is not None:
            return

        self._event_track_id = event_track_id
        self._last_detection_time = datetime.utcnow()
        clip_id = self._clip_id_for_event(detected_classes)
        self._event_clip_id = clip_id
        self._event_start_time = datetime.utcnow()
        self._detected_classes = set(detected_classes)
        self._class_detection_counts = {c: 0 for c in detected_classes}
        self._frame_metadata = []
        self._frame_count_at_stop = 0

        await self.video_recorder.start_recording(clip_id, fps=fps)
        if not self.video_recorder.current_recording:
            return
        self.video_recorder.current_recording["event_recording"] = True

        preroll = await self.ring_buffer.get_preroll_frames(self.preroll_seconds)
        if preroll:
            await self.video_recorder.add_frames_batch(preroll)
            logger.info(f"Event recording pre-roll: {len(preroll)} frames")

        if frame_info is not None:
            self._frame_metadata.append(frame_info)

        if self._on_recording_started:
            await self._on_recording_started(clip_id)
        logger.info(f"Started event recording: {clip_id}")

    def _save_metadata(self, clip_id: str, result: dict, event_track_id: Optional[int] = None) -> None:
        """Save event metadata as JSON next to the clip (e.g. clip_id_metadata.json)."""
        if self._event_start_time is None:
            return
        end_time = datetime.utcnow()
        duration = (end_time - self._event_start_time).total_seconds()
        metadata = {
            "event_name": clip_id,
            "track_id": event_track_id,
            "start_time": self._event_start_time.isoformat() + "Z",
            "end_time": end_time.isoformat() + "Z",
            "duration_seconds": round(duration, 2),
            "detected_classes": list(sorted(self._detected_classes)),
            "class_detection_counts": dict(self._class_detection_counts),
            "total_frames": result.get("frame_count", 0),
            "fps": result.get("declared_fps"),
            "video_file": f"{clip_id}.mp4",
            "frames": self._frame_metadata,
        }
        meta_path = self.video_recorder.clips_dir / f"{clip_id}_metadata.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved event metadata: {meta_path.name}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    async def stop_event_recording(self) -> Optional[dict]:
        """Stop current event recording and save metadata."""
        if not self.is_recording or self._event_clip_id is None:
            return None
        clip_id = self._event_clip_id
        event_track_id = self._event_track_id
        self._event_clip_id = None
        self._event_track_id = None
        self._last_detection_time = None
        rec = self.video_recorder.current_recording
        if rec:
            self._frame_count_at_stop = rec.get("frame_count", 0)
        result = await self.video_recorder.stop_recording()
        if result:
            self._save_metadata(clip_id, result, event_track_id=event_track_id)
            if self._on_recording_saved:
                await self._on_recording_saved(result)
        self._event_start_time = None
        self._detected_classes = set()
        self._class_detection_counts = {}
        self._frame_metadata = []
        logger.info(f"Stopped event recording: {clip_id}")
        return result

    def should_stop_recording(self) -> bool:
        """True when our single event_track_id has been absent for recording_buffer_seconds."""
        if not self.is_recording or self._last_detection_time is None:
            return False
        elapsed = (datetime.utcnow() - self._last_detection_time).total_seconds()
        return elapsed >= self.recording_buffer_seconds

    async def update(
        self,
        current_track_ids: Set[int],
        detected_classes: Set[str],
        frame_info: Optional[dict] = None,
        fps: float = 30.0,
    ) -> None:
        """
        Update state from current frame. Call from YOLO consumer.
        - Start: when at least one track_id is present; lock onto one (e.g. first) as event_track_id.
        - While recording: only refresh last_detection_time when event_track_id is in current_track_ids.
        - Stop: when that exact event_track_id has been absent for recording_buffer_seconds.
        """
        if not self.is_recording:
            # Start event when we see at least one tracked object; use a single ID
            if current_track_ids:
                event_track_id = min(current_track_ids)  # pick one deterministically
                await self.start_event_recording(
                    event_track_id=event_track_id,
                    detected_classes=detected_classes,
                    fps=fps,
                    frame_info=frame_info,
                )
        else:
            # Only refresh timer when our tracked object (exact ID) is present
            if self._event_track_id is not None and self._event_track_id in current_track_ids:
                self._last_detection_time = datetime.utcnow()
                self._detected_classes.update(detected_classes)
                for c in detected_classes:
                    self._class_detection_counts[c] = self._class_detection_counts.get(c, 0) + 1

        if self.is_recording and frame_info is not None:
            frame_info = dict(frame_info)
            frame_info["frame_number"] = len(self._frame_metadata) + 1
            self._frame_metadata.append(frame_info)

        if self.is_recording and self.should_stop_recording():
            await self.stop_event_recording()

    async def cleanup(self) -> None:
        if self.is_recording:
            await self.stop_event_recording()
