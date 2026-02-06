"""
Simple Recorder Server
FastAPI application with WebRTC streaming, manual recording with 5 s buffer,
event-based recording on YOLO detection (stops when no object +3.5s),
and clip management.
"""

import asyncio
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime as _dt
from pathlib import Path
from typing import Set, Optional

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from aiortc import RTCPeerConnection, RTCSessionDescription

from .webcam_capture import WebcamCapture
from .webcam_track import WebcamStreamTrack
from .ring_buffer import RingBuffer
from .video_recorder import VideoRecorder
from .event_video_saver import EventVideoSaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pcs: Set[RTCPeerConnection] = set()
webcam_capture: Optional[WebcamCapture] = None
ring_buffer: Optional[RingBuffer] = None
video_recorder: Optional[VideoRecorder] = None
event_video_saver: Optional[EventVideoSaver] = None
yolo_model = None  # lazily loaded

# Recording state
is_recording = False
recording_clip_id: Optional[str] = None
postroll_task: Optional[asyncio.Task] = None

# WebSocket connections for live status updates
status_websockets: Set[WebSocket] = set()

BUFFER_SECONDS = 5  # pre-roll and post-roll duration


def _new_clip_id() -> str:
    return _dt.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]


# ---------------------------------------------------------------------------
# Helpers – broadcast over WebSocket
# ---------------------------------------------------------------------------
async def broadcast_status(msg: dict):
    """Send a JSON message to every connected status WebSocket."""
    dead: Set[WebSocket] = set()
    for ws in status_websockets:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    status_websockets.difference_update(dead)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
def _run_yolo_track(model, frame):
    """Run YOLO track in sync context (for asyncio.to_thread)."""
    return model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global webcam_capture, ring_buffer, video_recorder, event_video_saver, yolo_model

    logger.info("Starting Simple Recorder ...")

    # Load YOLO for event-based recording (optional)
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov26n.pt")
        logger.info("YOLO model loaded for event recording")
    except Exception as e:
        logger.warning(f"YOLO not available (event recording disabled): {e}")
        yolo_model = None

    try:
        webcam_capture = WebcamCapture(width=640, height=480, fps=30)
        loop = asyncio.get_running_loop()
        webcam_capture.start(loop)

        # Wait for the capture loop to produce at least one measured FPS
        await asyncio.sleep(3)
        cam_fps = webcam_capture.measured_fps
        logger.info(f"Webcam capture started (measured fps={cam_fps:.1f})")

        ring_buffer = RingBuffer(duration_seconds=BUFFER_SECONDS, fps=cam_fps)
        video_recorder = VideoRecorder(fps=cam_fps)
        async def on_event_recording_saved(result):
            if result:
                await broadcast_status({
                    "type": "recording_saved",
                    "clip": _clip_dict(result["clip_id"]),
                })

        event_video_saver = EventVideoSaver(
            video_recorder=video_recorder,
            ring_buffer=ring_buffer,
            recording_buffer_seconds=3.5,
            preroll_seconds=BUFFER_SECONDS,
            on_recording_saved=on_event_recording_saved,
        )

        # Consumer: feed every frame into the ring buffer
        async def ring_buffer_consumer(frame, _ts):
            await ring_buffer.add_frame(frame)

        # Consumer: if recorder is active, pipe frames to it
        async def recorder_consumer(frame, _ts):
            if video_recorder.current_recording:
                await video_recorder.add_frame(frame)

        # Consumer: YOLO tracking + event-based recording (one active recording at a time)
        async def yolo_event_consumer(frame, _ts):
            if yolo_model is None or event_video_saver is None:
                return
            try:
                results = await asyncio.to_thread(_run_yolo_track, yolo_model, frame)
            except Exception as e:
                logger.debug(f"YOLO track error: {e}")
                return
            boxes = results[0].boxes
            detections_present = (
                boxes.id is not None and len(boxes.id) > 0
            )
            detected_classes = set()
            detections = []
            if boxes is not None:
                track_ids = boxes.id.cpu().numpy().tolist() if boxes.id is not None else []
                class_ids = boxes.cls.cpu().numpy().tolist()
                confidences = boxes.conf.cpu().numpy().tolist()
                bboxes = boxes.xyxy.cpu().numpy().tolist()
                for i, cid in enumerate(class_ids):
                    name = yolo_model.names[int(cid)]
                    detected_classes.add(name)
                    detections.append({
                        "track_id": int(track_ids[i]) if i < len(track_ids) else None,
                        "class_id": int(cid),
                        "class_name": name,
                        "confidence": round(confidences[i], 3),
                        "bbox": [round(x, 2) for x in bboxes[i]],
                    })
            frame_info = {
                "timestamp": _dt.utcnow().isoformat() + "Z",
                "num_detections": len(detections),
                "detections": detections,
            }
            fps = webcam_capture.measured_fps if webcam_capture else 30.0
            await event_video_saver.update(
                detections_present=detections_present,
                detected_classes=detected_classes,
                frame_info=frame_info,
                fps=fps,
            )

        webcam_capture.add_consumer(ring_buffer_consumer)
        webcam_capture.add_consumer(recorder_consumer)
        if yolo_model is not None:
            webcam_capture.add_consumer(yolo_event_consumer)
    except Exception as e:
        logger.warning(f"Webcam not available: {e}")
        webcam_capture = None
        event_video_saver = None
        ring_buffer = RingBuffer(duration_seconds=BUFFER_SECONDS, fps=30.0)
        video_recorder = VideoRecorder(fps=30.0)

    logger.info("Simple Recorder ready")
    yield

    # Shutdown
    logger.info("Shutting down ...")
    if event_video_saver:
        await event_video_saver.cleanup()
    if video_recorder and video_recorder.current_recording:
        await video_recorder.stop_recording()
    if webcam_capture:
        webcam_capture.stop()
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Simple Recorder", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WebRTC
# ---------------------------------------------------------------------------
@app.post("/offer")
async def offer(request: Request):
    """SDP offer/answer exchange for WebRTC."""
    params = await request.json()
    sdp = params.get("sdp")
    typ = params.get("type")
    if not sdp or typ != "offer":
        return JSONResponse(status_code=400, content={"error": "Invalid offer"})

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    if webcam_capture and webcam_capture.is_running:
        pc.addTrack(WebcamStreamTrack(webcam_capture))

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=typ))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(content={
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })


# ---------------------------------------------------------------------------
# Recording control
# ---------------------------------------------------------------------------
@app.post("/api/recording/start")
async def start_recording():
    """Start a new recording with 5 s pre-roll from the ring buffer."""
    global is_recording, recording_clip_id, postroll_task

    if is_recording:
        return JSONResponse(status_code=409, content={"error": "Already recording"})
    if event_video_saver and event_video_saver.is_recording:
        return JSONResponse(
            status_code=409,
            content={"error": "Event recording in progress; only one active recording at a time"},
        )

    # If a previous post-roll is still running, finalize it immediately
    if postroll_task and not postroll_task.done():
        postroll_task.cancel()
        if video_recorder.current_recording:
            await video_recorder.stop_recording()
        postroll_task = None

    clip_id = _new_clip_id()
    recording_clip_id = clip_id

    # Use current (smoothed) measured FPS so encoding rate matches delivery during this recording
    recording_fps = webcam_capture.measured_fps if webcam_capture else 30.0

    # Start the recorder and write pre-roll frames
    await video_recorder.start_recording(clip_id, fps=recording_fps)
    preroll = await ring_buffer.get_preroll_frames(BUFFER_SECONDS)
    if preroll:
        await video_recorder.add_frames_batch(preroll)
        logger.info(f"Wrote {len(preroll)} pre-roll frames")

    # #region agent log
    import json as _json, time as _time
    _DBG_LOG = "/media/siva/MEDIA/DEV/OBJECT-DETECTION/.cursor/debug.log"
    try:
        with open(_DBG_LOG, "a") as _f:
            _f.write(_json.dumps({"hypothesisId":"H3","location":"server.py:start_recording","message":"preroll_info","data":{"preroll_frame_count":len(preroll) if preroll else 0,"buffer_len":len(ring_buffer),"expected_for_5s":int(5*30)},"timestamp":int(_time.time()*1000),"sessionId":"debug-session"}) + "\n")
    except Exception:
        pass
    # #endregion

    is_recording = True

    await broadcast_status({"type": "recording_started", "clip_id": clip_id})
    return {"status": "recording", "clip_id": clip_id}


@app.post("/api/recording/stop")
async def stop_recording(postroll_seconds: int = Query(0, ge=0, description="Optional post-roll seconds; 0 = finalise immediately")):
    """Stop recording. Optional post-roll: if postroll_seconds > 0, capture that many seconds after stop then finalise. Default 0 = finalise immediately."""
    global is_recording, postroll_task

    if not is_recording:
        return JSONResponse(status_code=409, content={"error": "Not recording"})

    is_recording = False
    clip_id = recording_clip_id

    if postroll_seconds <= 0:
        # No post-roll: finalize immediately
        result = await video_recorder.stop_recording()
        if result:
            await broadcast_status({
                "type": "recording_saved",
                "clip": _clip_dict(result["clip_id"]),
            })
        return {"status": "saved", "clip_id": clip_id, "postroll_seconds": 0}

    # Post-roll: keep writing frames for postroll_seconds, then finalize
    await broadcast_status({
        "type": "postroll_started",
        "clip_id": clip_id,
        "seconds": postroll_seconds,
    })

    async def _postroll():
        try:
            await asyncio.sleep(postroll_seconds)
            result = await video_recorder.stop_recording()
            if result:
                await broadcast_status({
                    "type": "recording_saved",
                    "clip": _clip_dict(result["clip_id"]),
                })
        except asyncio.CancelledError:
            pass  # handled by caller

    postroll_task = asyncio.create_task(_postroll())

    return {"status": "stopping", "clip_id": clip_id, "postroll_seconds": postroll_seconds}


# ---------------------------------------------------------------------------
# Clips API
# ---------------------------------------------------------------------------
def _clip_dict(clip_id: str) -> dict:
    """Build a JSON-safe dict for a single clip."""
    for c in video_recorder.list_clips():
        if c["clip_id"] == clip_id:
            return {
                **c,
                "video_url": f"/api/clips/{c['clip_id']}/video",
                "thumbnail_url": (
                    f"/api/clips/{c['clip_id']}/thumbnail"
                    if c["has_thumbnail"] else None
                ),
            }
    return {"clip_id": clip_id}


@app.get("/api/clips")
async def list_clips():
    """List all recorded clips."""
    raw = video_recorder.list_clips()
    clips = []
    for c in raw:
        clips.append({
            **c,
            "video_url": f"/api/clips/{c['clip_id']}/video",
            "thumbnail_url": (
                f"/api/clips/{c['clip_id']}/thumbnail"
                if c["has_thumbnail"] else None
            ),
        })
    return {"clips": clips}


@app.get("/api/clips/{clip_id}/video")
async def get_clip_video(clip_id: str):
    """Serve an MP4 clip file."""
    path = video_recorder.clip_path(clip_id)
    if not path.exists() or path.stat().st_size == 0:
        return JSONResponse(status_code=404, content={"error": "Clip not found"})
    return FileResponse(str(path), media_type="video/mp4", filename=f"{clip_id}.mp4")


@app.get("/api/clips/{clip_id}/thumbnail")
async def get_clip_thumbnail(clip_id: str):
    """Serve a clip thumbnail."""
    path = video_recorder.thumb_path(clip_id)
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Thumbnail not found"})
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/api/clips/{clip_id}/metadata")
async def get_clip_metadata(clip_id: str):
    """Serve event metadata JSON for a clip (if present)."""
    meta_path = video_recorder.clips_dir / f"{clip_id}_metadata.json"
    if not meta_path.exists():
        return JSONResponse(status_code=404, content={"error": "Metadata not found"})
    return FileResponse(str(meta_path), media_type="application/json")


@app.delete("/api/clips/{clip_id}")
async def delete_clip(clip_id: str):
    """Delete a clip, its thumbnail, and metadata (if any)."""
    ok = video_recorder.delete_clip(clip_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "Clip not found"})
    meta_path = video_recorder.clips_dir / f"{clip_id}_metadata.json"
    if meta_path.exists():
        meta_path.unlink()
    await broadcast_status({"type": "clip_deleted", "clip_id": clip_id})
    return {"deleted": clip_id}


# ---------------------------------------------------------------------------
# Recording status
# ---------------------------------------------------------------------------
@app.get("/api/recording/status")
async def recording_status():
    """Return current recording state."""
    return {
        "is_recording": is_recording,
        "clip_id": recording_clip_id if is_recording else None,
        "postroll_active": (
            postroll_task is not None and not postroll_task.done()
        ) if postroll_task else False,
    }


# ---------------------------------------------------------------------------
# WebSocket for live status updates
# ---------------------------------------------------------------------------
@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    await websocket.accept()
    status_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        status_websockets.discard(websocket)


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the SPA."""
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    try:
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "webcam": webcam_capture is not None and webcam_capture.is_running,
        "connections": len(pcs),
    }
