"""Integration tests for pure offline VSR + SLM pipeline."""

import os
from pathlib import Path
import pytest
from backend.offline_slm import OfflineSLMEngine
from export.test_onnx_inference import StandaloneONNXVSR

GGUF_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
ONNX_PATH = "models/auto_avsr_visual_int8.onnx"


@pytest.mark.skipif(not os.path.exists(GGUF_PATH), reason="GGUF model not found")
def test_offline_slm_resolution():
    engine = OfflineSLMEngine(model_path=GGUF_PATH, n_ctx=512)
    input_text = "I WANT TO CO TO THE BARK TO PLAY PALL"
    corrected = engine.resolve(input_text)
    assert corrected, "Should return non-empty corrected string"
    # Should correct CO->go, BARK->park, PALL->ball
    lower = corrected.lower()
    assert "park" in lower or "ball" in lower or "go" in lower


@pytest.mark.skipif(
    not os.path.exists(GGUF_PATH) or not os.path.exists(ONNX_PATH),
    reason="ONNX or GGUF models not found",
)
def test_offline_pipeline_smoke():
    import numpy as np

    vsr = StandaloneONNXVSR(model_path=ONNX_PATH)
    slm = OfflineSLMEngine(model_path=GGUF_PATH, n_ctx=512)

    # 1. Run ONNX VSR on synthetic frames
    dummy_input = np.random.randn(1, 1, 37, 88, 88).astype(np.float32)
    raw_ctc = vsr.predict(dummy_input)

    # 2. Even if raw CTC is empty on random noise, verify SLM handles empty/short inputs safely
    res = slm.resolve(raw_ctc)
    assert isinstance(res, str)
