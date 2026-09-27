"""FastAPI application for Visual Speech Recognition with WebSocket streaming and REST APIs."""

import asyncio
import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.face_tracker import FaceTracker
from backend.llm_resolver import LLMResolver
from backend.resampler import TemporalResampler
from backend.vad import VisualVAD
from backend.vsr_engine import VSREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global singletons
vsr_engine: Optional[VSREngine] = None
llm_resolver: Optional[LLMResolver] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and clients on startup."""
    global vsr_engine, llm_resolver

    logger.info("Initializing VSR engine and Ollama resolver...")
    config_path = os.getenv("VSR_CONFIG_PATH", "backend/configs/LRS3_V_WER19.1.ini")
    vsr_engine = VSREngine(config_path=config_path)

    # Attempt to load model in background thread if weights exist
    if os.path.exists("benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth"):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, vsr_engine.load_model)
        except Exception as e:
            logger.warning("Could not pre-load VSR model at startup: %s", e)
    else:
        logger.warning(
            "Auto-AVSR weights not found at benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth. "
            "Run python scripts/download_models.py to download."
        )

    llm_resolver = LLMResolver()
    yield

    if llm_resolver:
        await llm_resolver.close()


app = FastAPI(
    title="Visual Speech Recognition Live Hub",
    description="Real-time in-the-wild silent speech recognition engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """System health, GPU status, and model availability."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    model_loaded = (vsr_engine is not None) and (vsr_engine.pipeline is not None)
    ollama_ready = await llm_resolver.is_available() if llm_resolver else False

    return {
        "status": "ok",
        "gpu": gpu_available,
        "gpu_name": gpu_name,
        "model_loaded": model_loaded,
        "ollama_available": ollama_ready,
    }


@app.post("/api/recognize-clip")
async def recognize_clip(file: UploadFile = File(...)):
    """Transcribe an uploaded video clip (.webm or .mp4) using the complete VSR + LLM pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    start_time = time.time()
    suffix = Path(file.filename).suffix or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        resampler = TemporalResampler(target_fps=25.0)
        face_tracker = FaceTracker()
        mouth_crops = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms = (frame_idx / fps) * 1000.0
            frame_idx += 1

            for resampled_frame in resampler.add_frame(frame, ts_ms):
                crop = face_tracker.process_frame(resampled_frame)
                if crop is not None:
                    mouth_crops.append(crop)

        cap.release()
        for resampled_frame in resampler.flush():
            crop = face_tracker.process_frame(resampled_frame)
            if crop is not None:
                mouth_crops.append(crop)

        if not mouth_crops:
            raise HTTPException(status_code=422, detail="No faces detected in video clip")

        # Run Auto-AVSR inference
        raw_text = vsr_engine.predict_frames(mouth_crops) if vsr_engine else ""

        # Run LLM homophene resolution
        corrected_text = ""
        if raw_text and llm_resolver:
            corrected_text = await llm_resolver.resolve_homophenes(raw_text)

        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "raw_text": raw_text,
            "corrected_text": corrected_text or raw_text,
            "latency_ms": round(latency_ms, 2),
            "num_frames": len(mouth_crops),
            "fps": 25.0,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Full-duplex WebSocket stream for real-time video frames and on-the-fly subtitles."""
    await websocket.accept()

    resampler = TemporalResampler(target_fps=25.0)
    face_tracker = FaceTracker()
    vad = VisualVAD()
    session_engine = VSREngine(config_path=vsr_engine.config_path if vsr_engine else "")
    if vsr_engine and vsr_engine.pipeline:
        session_engine.pipeline = vsr_engine.pipeline

    latest_corrected_text = ""
    last_llm_time = 0.0

    try:
        while True:
            # Expect binary image frame (JPEG/WebP) or text JSON control command
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                now_ms = time.time() * 1000.0

                # Decode JPEG/WebP frame from client
                np_arr = np.frombuffer(raw_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                emitted_frames = resampler.add_frame(frame, now_ms)
                for resampled_frame in emitted_frames:
                    crop = face_tracker.process_frame(resampled_frame)
                    face_detected = face_tracker.is_face_detected()
                    head_pose = face_tracker.get_head_pose()
                    landmarks = face_tracker.get_latest_mouth_landmarks()

                    if crop is not None:
                        is_speaking = vad.update(crop, landmarks, now_ms)
                        res = session_engine.process_sliding_window([crop])

                        # Trigger LLM homophene correction on completed phrase or periodically
                        if vad.is_phrase_complete() or (
                            res["full_transcript"]
                            and (time.time() - last_llm_time) > 1.2
                            and llm_resolver
                        ):
                            last_llm_time = time.time()
                            asyncio.create_task(
                                _async_resolve_and_send(
                                    websocket,
                                    res["full_transcript"],
                                    llm_resolver,
                                    face_detected,
                                    head_pose,
                                )
                            )

                        # Send instantaneous CTC subtitle update (<150ms)
                        await websocket.send_json({
                            "type": "subtitle_tier1",
                            "raw_chunk": res["raw_chunk"],
                            "full_raw": res["full_transcript"],
                            "committed_text": res["committed_transcript"],
                            "speaking": is_speaking,
                            "face_detected": face_detected,
                            "head_pose": list(head_pose),
                        })
                    else:
                        await websocket.send_json({
                            "type": "status",
                            "face_detected": False,
                            "head_pose": [0.0, 0.0, 0.0],
                        })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error("WebSocket stream error: %s", e)


async def _async_resolve_and_send(
    websocket: WebSocket,
    raw_text: str,
    resolver: LLMResolver,
    face_detected: bool,
    head_pose: tuple,
):
    """Background helper to resolve homophenes without blocking frame streaming."""
    try:
        corrected = await resolver.resolve_homophenes(raw_text)
        if corrected:
            await websocket.send_json({
                "type": "subtitle_tier2",
                "corrected_text": corrected,
                "face_detected": face_detected,
                "head_pose": list(head_pose),
            })
    except Exception as e:
        logger.debug("Background homophene send exception: %s", e)


# Mount frontend static directory if exists
frontend_dir = Path(__file__).resolve().parent.parent / "frontend_web"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(frontend_dir / "index.html")
