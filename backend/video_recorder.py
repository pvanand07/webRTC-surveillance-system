"""
Video Recorder - Records clips to disk with H.264 encoding via ffmpeg.
Generates thumbnails for each clip.
"""

from datetime import datetime
from typing import Optional, List
from pathlib import Path
import subprocess
import shutil
import asyncio
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def _ffmpeg_available() -> bool:
    """Check if ffmpeg with libx264 is available."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return "libx264" in (out.stdout or "") or "libx264" in (out.stderr or "")
    except Exception:
        return False


class VideoRecorder:
    """Records video clips with ffmpeg libx264 and generates thumbnails."""

    def __init__(
        self,
        clips_dir: str = None,
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
    ):
        if clips_dir is None:
            clips_dir = str(Path(__file__).parent.parent / "clips")

        self.clips_dir = Path(clips_dir)
        self.fps = fps
        self.width = width
        self.height = height

        self.current_recording: Optional[dict] = None
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._frames_buffer: List[np.ndarray] = []
        self._has_ffmpeg = _ffmpeg_available()

        self.clips_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"VideoRecorder ready ({width}x{height} @ {fps}fps, "
            f"encoder={'libx264' if self._has_ffmpeg else 'UNAVAILABLE'})"
        )

    # ---- path helpers ----

    def clip_path(self, clip_id: str) -> Path:
        return self.clips_dir / f"{clip_id}.mp4"

    def thumb_path(self, clip_id: str) -> Path:
        return self.clips_dir / f"{clip_id}_thumb.jpg"

    # ---- recording lifecycle ----

    async def start_recording(self, clip_id: str, fps: Optional[float] = None) -> dict:
        """Open a new recording session. Returns recording metadata dict.
        fps: frame rate for this recording (default: self.fps). Use actual measured FPS when provided.
        """
        clip = str(self.clip_path(clip_id))
        recording_fps = fps if fps is not None and fps > 0 else self.fps

        self.current_recording = {
            "clip_id": clip_id,
            "clip_path": clip,
            "thumb_path": str(self.thumb_path(clip_id)),
            "start_time": datetime.utcnow(),
            "frame_count": 0,
            "declared_fps": recording_fps,
        }

        if self._has_ffmpeg:
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(recording_fps),
                "-i", "pipe:0",
                "-c:v", "libx264", "-preset", "fast",
                "-profile:v", "baseline", "-level", "3.1",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                clip,
            ]
            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            logger.info(f"Recording started: {clip_id}")
        else:
            logger.error("ffmpeg not available – cannot record")

        self._frames_buffer = []
        return self.current_recording

    async def add_frame(self, frame: np.ndarray) -> None:
        """Write a BGR numpy frame to the active recording."""
        if not self.current_recording or not self._ffmpeg_process:
            return
        try:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            self._ffmpeg_process.stdin.write(frame.tobytes())

            # Buffer a few frames for thumbnail
            if self.current_recording["frame_count"] < 10:
                self._frames_buffer.append(frame.copy())

            self.current_recording["frame_count"] += 1
        except BrokenPipeError:
            logger.warning("ffmpeg pipe closed early")
        except Exception as e:
            logger.error(f"Error writing frame: {e}")

    async def add_frames_batch(self, frames: List[np.ndarray]) -> None:
        """Write a batch of pre-roll frames."""
        for f in frames:
            await self.add_frame(f)

    async def stop_recording(self) -> Optional[dict]:
        """Finalize the current recording, generate thumbnail, return metadata."""
        if not self.current_recording:
            return None

        duration = (datetime.utcnow() - self.current_recording["start_time"]).total_seconds()
        # #region agent log
        import json as _json, time as _time
        _DBG_LOG = "/media/siva/MEDIA/DEV/OBJECT-DETECTION/.cursor/debug.log"
        _fc = self.current_recording["frame_count"]
        _effective_fps = _fc / duration if duration > 0 else 0
        _declared = self.current_recording.get("declared_fps", self.fps)
        try:
            with open(_DBG_LOG, "a") as _f:
                _f.write(_json.dumps({"hypothesisId":"H1_H2","location":"video_recorder.py:stop_recording","message":"recording_stats","data":{"frame_count":_fc,"duration_sec":round(duration,2),"effective_fps":round(_effective_fps,2),"declared_fps":_declared},"timestamp":int(_time.time()*1000),"sessionId":"debug-session"}) + "\n")
        except Exception:
            pass
        # #endregion

        # Close ffmpeg
        if self._ffmpeg_process:
            proc = self._ffmpeg_process
            self._ffmpeg_process = None
            try:
                proc.stdin.close()
                await asyncio.to_thread(lambda: proc.wait(timeout=30))
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg timed out, killing")
                proc.kill()
            except Exception as e:
                logger.warning(f"ffmpeg finalize: {e}")
                try:
                    proc.kill()
                except Exception:
                    pass

        # Generate thumbnail
        thumb = self.current_recording["thumb_path"]
        if self._frames_buffer:
            self._generate_thumbnail(self._frames_buffer, thumb)
        else:
            self._placeholder_thumbnail(thumb)

        result = {
            **self.current_recording,
            "duration": round(duration, 2),
        }

        logger.info(
            f"Recording finalized: {result['clip_id']} "
            f"({result['duration']}s, {result['frame_count']} frames)"
        )

        self.current_recording = None
        self._frames_buffer = []
        return result

    # ---- thumbnails ----

    def _generate_thumbnail(self, frames: List[np.ndarray], path: str) -> None:
        try:
            idx = len(frames) // 2
            thumb = cv2.resize(frames[idx], (320, 240))
            cv2.imwrite(path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception as e:
            logger.error(f"Thumbnail error: {e}")
            self._placeholder_thumbnail(path)

    def _placeholder_thumbnail(self, path: str) -> None:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, "No Preview", (70, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ---- helpers ----

    def clip_exists(self, clip_id: str) -> bool:
        p = self.clip_path(clip_id)
        return p.exists() and p.stat().st_size > 0

    def delete_clip(self, clip_id: str) -> bool:
        deleted = False
        for p in (self.clip_path(clip_id), self.thumb_path(clip_id)):
            if p.exists():
                p.unlink()
                deleted = True
        return deleted

    def list_clips(self) -> List[dict]:
        """Scan the clips directory and return metadata for every .mp4 file."""
        clips = []
        for mp4 in sorted(self.clips_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            clip_id = mp4.stem
            thumb = self.thumb_path(clip_id)
            stat = mp4.stat()
            clips.append({
                "clip_id": clip_id,
                "filename": mp4.name,
                "size_bytes": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "has_thumbnail": thumb.exists(),
            })
        return clips
