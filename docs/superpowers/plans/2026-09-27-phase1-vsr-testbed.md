# Phase 1: Rapid Prototyping & Field Validation Testbed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, headless Visual Speech Recognition (VSR) testbed with live on-the-fly subtitles accessible via desktop and mobile phone browser over local Wi-Fi, powered by Auto-AVSR (`LRS3_V_WER19.1`), MediaPipe canonical affine multi-angle normalization, and Ollama Qwen 2.5 homophene resolution.

**Architecture:** A Python FastAPI backend running on the local RTX 5060 Ti GPU serves a full-duplex WebSocket stream (`/ws/stream`) and REST API. Incoming camera frames from the mobile/desktop web client are resampled to exact 25.0 FPS, tracked with MediaPipe Face Mesh, warped via canonical affine transformation to 96×96 grayscale mouth crops (handling head yaw/pitch up to ±45°), and fed to a 1.5s sliding-window Conformer engine. Visual VAD boundaries trigger async Ollama Qwen homophene correction for on-the-fly subtitles.

**Tech Stack:** Python 3.11, `uv`, PyTorch (CUDA 12/13), Torchvision, Torchaudio, OpenCV, MediaPipe, FastAPI, Uvicorn, WebSockets, Ollama (`qwen2.5:3b` / `7b`), Vanilla HTML5/CSS/JavaScript (PWA).

## Global Constraints

* Strict 25.0 FPS input into Auto-AVSR Conformer (resample variable camera FPS).
* Canonical affine warping maps mouth landmarks to standard 96×96 coordinates.
* Headless backend: No GUI windows (`cv2.imshow`), no desktop keyboard hooks (`pynput`).
* All code paths must be unit-tested using `pytest`.
* Frequent commits at each task cornerstone, and git push at key milestones.

---

### Task 1: Environment Setup & Model Artifacts Acquisition

**Files:**
- Create: `pyproject.toml`
- Create: `backend/configs/default.ini`
- Create: `scripts/download_models.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Produces: `uv` virtual environment with PyTorch CUDA, Auto-AVSR dependencies, and downloaded `LRS3_V_WER19.1` PyTorch checkpoint in `models/`.

- [ ] **Step 1: Write environment and dependency verification test**

```python
# tests/test_environment.py
import torch
import cv2
import mediapipe
import fastapi

def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA must be available for RTX 5060 Ti"
    assert torch.cuda.device_count() >= 1

def test_dependencies_importable():
    import numpy
    import scipy
    import av
    assert True
```

- [ ] **Step 2: Create `pyproject.toml` and configure `uv` virtual environment**

```toml
[project]
name = "visual-speech-recognition"
version = "0.1.0"
description = "In-the-wild Visual Speech Recognition testbed and mobile pipeline"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "torchaudio>=2.2.0",
    "opencv-python-headless>=4.9.0",
    "mediapipe>=0.10.9",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.28.0",
    "websockets>=12.0",
    "python-multipart>=0.0.9",
    "httpx>=0.27.0",
    "ollama>=0.1.7",
    "pydantic>=2.6.0",
    "scipy>=1.12.0",
    "av>=11.0.0",
    "sentencepiece>=0.2.0",
    "hydra-core>=1.3.2",
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

- [ ] **Step 3: Create model download script `scripts/download_models.py`**

Script downloads the `LRS3_V_WER19.1` visual checkpoint, language dictionary / SentencePiece subword model, and default Auto-AVSR configs from Hugging Face Hub (`mpc001` / `Amanvir`) into `models/`.

- [ ] **Step 4: Execute model download and run environment test**

Run: `uv run pytest tests/test_environment.py -v`  
Expected: PASS

- [ ] **Step 5: Commit cornerstone**

```bash
git add pyproject.toml scripts/download_models.py tests/test_environment.py backend/configs/
git commit -m "chore: setup uv environment, dependencies, and model download script"
```

---

### Task 2: Strict 25.0 FPS Temporal Resampler

**Files:**
- Create: `backend/resampler.py`
- Test: `tests/test_resampler.py`

**Interfaces:**
- Produces: `TemporalResampler` class:
  ```python
  class TemporalResampler:
      def __init__(self, target_fps: float = 25.0) -> None: ...
      def add_frame(self, frame: np.ndarray, timestamp_ms: float) -> list[np.ndarray]: ...
      def flush(self) -> list[np.ndarray]: ...
      def reset(self) -> None: ...
  ```

- [ ] **Step 1: Write failing test for 25.0 FPS resampling from 30 FPS input**

```python
# tests/test_resampler.py
import numpy as np
from backend.resampler import TemporalResampler

def test_resample_30fps_to_25fps():
    resampler = TemporalResampler(target_fps=25.0)
    output_frames = []
    # Simulate 30 frames spanning 1000ms (~33.3ms intervals)
    for i in range(30):
        frame = np.full((96, 96, 3), i, dtype=np.uint8)
        ts = i * (1000.0 / 30.0)
        output_frames.extend(resampler.add_frame(frame, ts))
    output_frames.extend(resampler.flush())
    # Exactly 1 second at 25 FPS must yield 25 frames
    assert len(output_frames) == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_resampler.py -v`  
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `backend/resampler.py`**

Implements monotonic 40ms timeline grid tracking with nearest-timestamp frame selection and buffer management.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_resampler.py -v`  
Expected: PASS

- [ ] **Step 5: Commit cornerstone**

```bash
git add backend/resampler.py tests/test_resampler.py
git commit -m "feat(vision): implement strict 25.0 FPS timestamp-based temporal resampler"
```

---

### Task 3: Robust Face Mesh Tracker & Canonical 2D Affine Warper

**Files:**
- Create: `backend/face_tracker.py`
- Test: `tests/test_face_tracker.py`

**Interfaces:**
- Produces: `FaceTracker` class:
  ```python
  class FaceTracker:
      def __init__(self, ema_alpha: float = 0.7) -> None: ...
      def process_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]: ...
      # Returns 96x96 grayscale numpy array normalized to [0, 255]
      def get_head_pose(self) -> tuple[float, float, float]: ... # yaw, pitch, roll
      def is_face_detected(self) -> bool: ...
  ```

- [ ] **Step 1: Write failing test for face detection and affine mouth normalization**

```python
# tests/test_face_tracker.py
import numpy as np
import cv2
from backend.face_tracker import FaceTracker

def test_face_tracker_output_shape():
    tracker = FaceTracker()
    # Create synthetic test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Tracker should return None when no face is present
    result = tracker.process_frame(frame)
    assert result is None
    assert not tracker.is_face_detected()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_face_tracker.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `backend/face_tracker.py`**

- Uses MediaPipe FaceMesh to extract 468 landmarks.
- Extracts key mouth anchors (lip corners 61 & 291, top lip 0, bottom lip 17).
- Computes canonical 2D affine transformation mapping mouth anchors to canonical frontal coordinates.
- Warps and crops to 96×96 grayscale image with histogram equalization/normalization.
- Smooths landmarks with Exponential Moving Average (`ema_alpha = 0.7`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_face_tracker.py -v`  
Expected: PASS

- [ ] **Step 5: Commit cornerstone**

```bash
git add backend/face_tracker.py tests/test_face_tracker.py
git commit -m "feat(vision): implement FaceMesh tracking and canonical 96x96 affine mouth warper"
```

---

### Task 4: Visual Voice Activity Detection (V-VAD)

**Files:**
- Create: `backend/vad.py`
- Test: `tests/test_vad.py`

**Interfaces:**
- Produces: `VisualVAD` class:
  ```python
  class VisualVAD:
      def __init__(self, lar_threshold: float = 0.15, motion_threshold: float = 2.0, silence_duration_ms: float = 400.0) -> None: ...
      def update(self, mouth_crop_96x96: np.ndarray, landmarks_mouth: np.ndarray, timestamp_ms: float) -> bool: ...
      def is_speaking(self) -> bool: ...
      def is_phrase_complete(self) -> bool: ...
      def reset(self) -> None: ...
  ```

- [ ] **Step 1: Write failing test for V-VAD speech detection and silence closure**

```python
# tests/test_vad.py
import numpy as np
from backend.vad import VisualVAD

def test_vad_silence_detection():
    vad = VisualVAD(silence_duration_ms=400.0)
    empty_crop = np.zeros((96, 96), dtype=np.uint8)
    dummy_landmarks = np.zeros((4, 2)) # closed mouth
    # Feed 500ms of identical frames
    for t in range(0, 500, 40):
        vad.update(empty_crop, dummy_landmarks, float(t))
    assert not vad.is_speaking()
    assert vad.is_phrase_complete()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vad.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `backend/vad.py`**

Calculates Lip Aspect Ratio (LAR) and inter-frame absolute pixel delta, detecting onset and offset of mouth movements with debouncing.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vad.py -v`  
Expected: PASS

- [ ] **Step 5: Commit cornerstone**

```bash
git add backend/vad.py tests/test_vad.py
git commit -m "feat(vad): implement Visual Voice Activity Detection with LAR and kinematic motion"
```

---

### Task 5: Auto-AVSR Inference Engine & Sliding-Window Token Merger

**Files:**
- Create: `backend/vsr_engine.py`
- Test: `tests/test_vsr_engine.py`

**Interfaces:**
- Produces: `VSREngine` class:
  ```python
  class VSREngine:
      def __init__(self, config_path: str, model_path: str, device: str = "cuda:0") -> None: ...
      def predict_window(self, mouth_frames_tensor: torch.Tensor) -> str: ... # (1, 1, T, 96, 96)
      def process_sliding_window(self, new_frames_25fps: list[np.ndarray]) -> dict: ...
      # Returns {"raw_token": str, "committed_text": str, "tentative_text": str}
  ```

- [ ] **Step 1: Write failing test for sliding-window token merger**

```python
# tests/test_vsr_engine.py
import pytest
from backend.vsr_engine import merge_sliding_tokens

def test_merge_sliding_tokens():
    prev_text = "HELLO HOW"
    new_chunk = "HOW ARE YOU"
    merged, committed = merge_sliding_tokens(prev_text, new_chunk)
    assert "HELLO HOW ARE YOU" in merged
    assert "HELLO HOW" in committed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vsr_engine.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `backend/vsr_engine.py`**

- Loads Auto-AVSR PyTorch Conformer backbone (`LRS3_V_WER19.1`) on CUDA.
- Maintains rolling buffer of 37 frames (1.5s) stepping by 12 frames (0.5s).
- Runs CTC decoding to produce visual subwords.
- Implements `merge_sliding_tokens` prefix alignment to stitch continuous subtitles without duplicates.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vsr_engine.py -v`  
Expected: PASS

- [ ] **Step 5: Commit cornerstone**

```bash
git add backend/vsr_engine.py tests/test_vsr_engine.py
git commit -m "feat(vsr): implement Auto-AVSR inference engine with 1.5s sliding-window merger"
```

---

### Task 6: Async Ollama Homophene Resolver

**Files:**
- Create: `backend/llm_resolver.py`
- Test: `tests/test_llm_resolver.py`

**Interfaces:**
- Produces: `LLMResolver` class:
  ```python
  class LLMResolver:
      def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:3b") -> None: ...
      async def resolve_homophenes(self, raw_vsr_text: str) -> str: ...
      async def is_available(self) -> bool: ...
  ```

- [ ] **Step 1: Write test with mocked Ollama API response for homophene correction**

```python
# tests/test_llm_resolver.py
import pytest
from unittest.mock import AsyncMock, patch
from backend.llm_resolver import LLMResolver

@pytest.mark.asyncio
async def test_homophene_resolution():
    resolver = LLMResolver()
    with patch.object(resolver.client, 'chat', new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = {
            'message': {'content': 'I want to go to the park.'}
        }
        result = await resolver.resolve_homophenes("I WANT TO CO TO THE BARK")
        assert result == "I want to go to the park."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_resolver.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `backend/llm_resolver.py`**

Configures `httpx` async client to communicate with Ollama, using few-shot prompt instructions tailored for homophene correction, syntax repair, and anti-hallucination guardrails.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_resolver.py -v`  
Expected: PASS

- [ ] **Step 5: Commit cornerstone**

```bash
git add backend/llm_resolver.py tests/test_llm_resolver.py
git commit -m "feat(llm): implement async Ollama Qwen homophene resolver"
```

---

### Task 7: FastAPI Server with WebSocket Stream & REST Endpoints

**Files:**
- Create: `backend/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Exposes:
  - `GET /api/health` -> `{"status": "ok", "gpu": bool, "model_loaded": bool, "ollama": bool}`
  - `POST /api/recognize-clip` -> accepts multipart video (`.webm`, `.mp4`), returns `{"raw_text": str, "corrected_text": str, "latency_ms": float}`
  - `WebSocket /ws/stream` -> bidirectional full-duplex binary frame streaming and real-time subtitle delivery.

- [ ] **Step 1: Write API tests using FastAPI TestClient**

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `backend/app.py`**

- Assembles `TemporalResampler`, `FaceTracker`, `VisualVAD`, `VSREngine`, and `LLMResolver`.
- Adds CORS middleware for local network access (`0.0.0.0:8000`).
- Implements WebSocket handler decoding binary frames, feeding sliding window, and streaming back dual-tier subtitle JSON.
- Serves static frontend files from `frontend_web/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py -v`  
Expected: PASS

- [ ] **Step 5: Commit cornerstone**

```bash
git add backend/app.py tests/test_api.py
git commit -m "feat(api): implement FastAPI server with full-duplex WebSocket stream and REST"
```

---

### Task 8: Mobile-Friendly Web Client HUD

**Files:**
- Create: `frontend_web/index.html`
- Create: `frontend_web/style.css`
- Create: `frontend_web/app.js`

**Features:**
- Responsive mobile & desktop layout.
- Camera access with front/rear camera toggle (`facingMode: "user" | "environment"`).
- Oval face alignment reticle with real-time status (Green: Face Detected, Red: Face Lost).
- Two interaction modes:
  1. Live Subtitles (Continuous sliding window stream).
  2. Push-to-Talk (Hold button / Spacebar / Screen tap).
- Dual-tier subtitle display:
  - Instant raw visual words (<150ms).
  - Smooth animated typing of refined English sentences.
- Live latency and FPS diagnostics overlay.

- [ ] **Step 1: Build semantic HTML5 structure `frontend_web/index.html`**
- [ ] **Step 2: Build high-contrast, dark-mode styling `frontend_web/style.css`**
- [ ] **Step 3: Build camera capture & WebSocket streaming engine `frontend_web/app.js`**
- [ ] **Step 4: Verify UI locally in browser via Antigravity browser subagent**
- [ ] **Step 5: Commit cornerstone**

```bash
git add frontend_web/
git commit -m "feat(ui): create mobile-first responsive web client with camera HUD and subtitle renderer"
```

---

### Task 9: End-to-End Pipeline Integration & Benchmark Verification

**Files:**
- Create: `tests/test_e2e_pipeline.py`
- Create: `scripts/run_benchmark.py`

**Validation Criteria:**
- Feed a sample video clip through the full pipeline:
  `Raw Frames -> 25 FPS Resampler -> Face Mesh Affine Warper -> Auto-AVSR Conformer -> Ollama LLM`.
- Verify output text accuracy and total latency under 500ms on RTX 5060 Ti.
- Push all changes to remote GitHub repository.

- [ ] **Step 1: Write end-to-end integration test**
- [ ] **Step 2: Run end-to-end test and benchmark**

Run: `uv run pytest tests/test_e2e_pipeline.py -v`  
Expected: PASS

- [ ] **Step 3: Commit and Push Milestone**

```bash
git add tests/test_e2e_pipeline.py scripts/run_benchmark.py
git commit -m "test: add end-to-end pipeline verification and benchmark script"
git push origin main
```
