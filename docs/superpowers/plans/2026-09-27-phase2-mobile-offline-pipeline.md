# Phase 2 & 3: Offline Mobile Pipeline & Flutter Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 100% offline, cross-platform mobile visual speech recognition system. The solution pairs our quantized Auto-AVSR visual Conformer (`models/auto_avsr_visual_int8.onnx`, 302MB, 88ms latency) with an on-device quantized SLM (`qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491MB) and delivers a complete Flutter mobile application (`mobile_app/`) running with zero internet connectivity.

**Architecture:**
1. **On-Device Vision (Dart / MLKit):** Camera stream resampled to strict 25.0 FPS; FaceMesh landmarks extract lip anchors; 2D affine transformation warps mouth to canonical 96×96 grayscale images.
2. **On-Device VSR Inference (ONNX Runtime Mobile):** The 96×96 crops are center-cropped to 88×88, normalized, and fed in 1.5s sliding windows (37 frames) to `auto_avsr_visual_int8.onnx`. A fast Dart CTC greedy decoder collapses repeated frames and maps token IDs via `unigram5000_units.txt` to raw visual words.
3. **On-Device Homophene Resolver (`llama.cpp` / GGUF):** When Visual VAD detects phrase boundaries or pause intervals, the raw CTC text is sent to the local Qwen 2.5 0.5B SLM to correct viseme ambiguities (e.g., *bark* $\rightarrow$ *park*, *pall* $\rightarrow$ *ball*).
4. **Mobile HUD (Flutter):** Full-screen camera viewfinder with real-time face alignment oval and floating dual-tier subtitles.

**Tech Stack:** Dart, Flutter, `onnxruntime_flutter`, `google_mlkit_face_mesh_detection`, `llama_cpp_dart`, Python 3.11 (validation), `pytest`.

---

## File Structure

```
visual-speech-recognition/
├── models/
│   ├── auto_avsr_visual_int8.onnx         # 302 MB Quantized VSR backbone (Done)
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf  # 491 MB Quantized SLM
├── pipelines/tokens/
│   └── unigram5000_units.txt              # 45 KB CTC subword vocabulary
├── scripts/
│   ├── download_slm.py                    # Downloads Qwen2.5 0.5B GGUF
│   └── benchmark_offline_pipeline.py      # End-to-end benchmark of ONNX + GGUF
├── backend/
│   └── offline_slm.py                     # Python GGUF runner for validation
├── tests/
│   ├── test_download_slm.py               # Verifies GGUF checksum and integrity
│   └── test_offline_pipeline.py           # End-to-end offline VSR + SLM test
└── mobile_app/
    ├── pubspec.yaml                       # Flutter dependencies & assets
    ├── assets/
    │   ├── models/auto_avsr_visual_int8.onnx
    │   ├── models/qwen2.5-0.5b-instruct-q4_k_m.gguf
    │   └── tokens/unigram5000_units.txt
    ├── lib/
    │   ├── main.dart                      # Flutter app entry point
    │   ├── models/
    │   │   ├── head_pose.dart             # Pitch, yaw, roll data model
    │   │   └── subtitle_event.dart        # Tier 1 raw and Tier 2 refined models
    │   ├── services/
    │   │   ├── temporal_resampler.dart    # Strict 25.0 FPS timeline resampler
    │   │   ├── affine_mouth_warper.dart   # Canonical 96x96 affine transform
    │   │   ├── visual_vad.dart            # LAR and motion delta voice activity
    │   │   ├── vsr_onnx_engine.dart       # ONNX Runtime mobile inference & CTC
    │   │   └── offline_slm_resolver.dart  # llama.cpp GGUF homophene resolver
    │   └── ui/
    │       ├── viewfinder_hud.dart        # Camera preview + face guide oval
    │       └── subtitle_overlay.dart      # Real-time animated subtitle HUD
    └── test/
        ├── temporal_resampler_test.dart   # Dart 25 FPS resampler unit tests
        ├── ctc_decoder_test.dart          # Dart CTC greedy decoding unit tests
        └── token_merger_test.dart         # Dart sliding window token merger tests
```

---

### Task 1: Offline Mobile SLM Acquisition & Packaging

**Files:**
- Create: `scripts/download_slm.py`
- Test: `tests/test_download_slm.py`

**Interfaces:**
- Produces: `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` (491 MB) verified by file size and GGUF header magic.

- [ ] **Step 1: Write verification test for SLM download**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `scripts/download_slm.py` with progress tracking**
- [ ] **Step 4: Execute download and verify test passes**
- [ ] **Step 5: Commit cornerstone**

```bash
git add scripts/download_slm.py tests/test_download_slm.py
git commit -m "feat(models): add automated downloader and verification for Qwen2.5 0.5B GGUF"
```

---

### Task 2: Pure Offline End-to-End Python Validation

**Files:**
- Create: `backend/offline_slm.py`
- Create: `scripts/benchmark_offline_pipeline.py`
- Test: `tests/test_offline_pipeline.py`

**Interfaces:**
- Validates the complete zero-network chain:
  `Mouth Frames -> StandaloneONNXVSR -> CTC Greedy Decode -> Offline GGUF SLM`.
- Measures total latency of ONNX (~88ms) + SLM (~150-250ms) = Sub-350ms offline response.

- [ ] **Step 1: Write integration test for pure offline VSR + SLM**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `backend/offline_slm.py`**
- [ ] **Step 4: Run tests and offline benchmark**
- [ ] **Step 5: Commit cornerstone**

```bash
git add backend/offline_slm.py scripts/benchmark_offline_pipeline.py tests/test_offline_pipeline.py
git commit -m "feat(offline): validate zero-network VSR ONNX and GGUF SLM pipeline"
```

---

### Task 3: Flutter Mobile Core Services (Dart)

**Files:**
- Create: `mobile_app/pubspec.yaml`
- Create: `mobile_app/lib/services/temporal_resampler.dart`
- Create: `mobile_app/lib/services/visual_vad.dart`
- Create: `mobile_app/lib/services/affine_mouth_warper.dart`
- Test: `mobile_app/test/temporal_resampler_test.dart`

**Interfaces:**
- Port strict 25.0 FPS monotonic resampling to Dart.
- Port LAR and motion delta Visual VAD to Dart.
- Port canonical affine 96×96 mouth normalization to Dart.

- [ ] **Step 1: Write Dart unit tests for resampler and VAD**
- [ ] **Step 2: Implement `mobile_app/lib/services/temporal_resampler.dart`**
- [ ] **Step 3: Implement `mobile_app/lib/services/visual_vad.dart`**
- [ ] **Step 4: Implement `mobile_app/lib/services/affine_mouth_warper.dart`**
- [ ] **Step 5: Commit cornerstone**

```bash
git add mobile_app/lib/services/ mobile_app/test/
git commit -m "feat(flutter): implement 25 FPS resampler, VAD, and affine warper in Dart"
```

---

### Task 4: Flutter On-Device Inference Engines (ONNX & llama.cpp)

**Files:**
- Create: `mobile_app/lib/services/vsr_onnx_engine.dart`
- Create: `mobile_app/lib/services/offline_slm_resolver.dart`
- Test: `mobile_app/test/ctc_decoder_test.dart`
- Test: `mobile_app/test/token_merger_test.dart`

**Interfaces:**
- `VsrOnnxEngine`:
  - Loads `auto_avsr_visual_int8.onnx`.
  - Normalizes input frames (CenterCrop 88×88, scale 1/255, Normalize 0.421/0.165).
  - Evaluates ONNX session on NPU / CPU.
  - Implements greedy CTC decoding and sliding window token merger.
- `OfflineSlmResolver`:
  - Loads `qwen2.5-0.5b-instruct-q4_k_m.gguf` via `llama_cpp_dart`.
  - Runs few-shot homophene correction prompt with temperature 0.1.

- [ ] **Step 1: Write Dart unit tests for CTC decoder and token merger**
- [ ] **Step 2: Implement `mobile_app/lib/services/vsr_onnx_engine.dart`**
- [ ] **Step 3: Implement `mobile_app/lib/services/offline_slm_resolver.dart`**
- [ ] **Step 4: Commit cornerstone**

```bash
git add mobile_app/lib/services/vsr_onnx_engine.dart mobile_app/lib/services/offline_slm_resolver.dart
git commit -m "feat(flutter): implement on-device ONNX VSR and llama.cpp SLM services"
```

---

### Task 5: Flutter UI HUD & Floating Live Subtitles

**Files:**
- Create: `mobile_app/lib/main.dart`
- Create: `mobile_app/lib/models/head_pose.dart`
- Create: `mobile_app/lib/models/subtitle_event.dart`
- Create: `mobile_app/lib/ui/viewfinder_hud.dart`
- Create: `mobile_app/lib/ui/subtitle_overlay.dart`

**Features:**
- Full-screen camera viewfinder with front/back camera switch.
- SVG/CustomPainter oval face alignment guide and mouth reticle.
- Diagnostics overlay: real-time FPS, inference latency, face lock status.
- Dual-tier subtitle widget:
  - Instant raw visual words (<100ms).
  - Smooth typing animation of refined English sentences.

- [ ] **Step 1: Implement data models and subtitle event streams**
- [ ] **Step 2: Implement `viewfinder_hud.dart` with custom painter reticle**
- [ ] **Step 3: Implement `subtitle_overlay.dart`**
- [ ] **Step 4: Wire all components into `main.dart`**
- [ ] **Step 5: Commit and push milestone**

```bash
git add mobile_app/
git commit -m "feat(flutter): complete offline Visual Speech Recognition mobile app"
git push origin feat/vsr-testbed
```
