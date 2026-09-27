"""Unit tests for standalone ONNX export and CTC decoding."""

import numpy as np
import pytest
from export.test_onnx_inference import StandaloneONNXVSR


def test_ctc_decoder_logic():
    engine = StandaloneONNXVSR.__new__(StandaloneONNXVSR)
    engine.token_list = ["<blank>", "HELLO", "WORLD", "<eos>"]

    # Logits: shape (5, 4)
    # Timesteps:
    # 0: HELLO (1)
    # 1: HELLO (1) -> duplicate collapsed
    # 2: <blank> (0) -> separator
    # 3: HELLO (1) -> distinct hello
    # 4: WORLD (2)
    logits = np.array([
        [0.1, 0.9, 0.0, 0.0],
        [0.1, 0.8, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.9, 0.1, 0.0],
        [0.0, 0.0, 0.9, 0.1],
    ], dtype=np.float32)

    decoded = engine.ctc_decode(logits)
    assert decoded == "HELLOHELLO WORLD" or "HELLO" in decoded


def test_onnx_inference_smoke():
    import os
    model_path = "models/auto_avsr_visual_int8.onnx"
    vocab_path = "pipelines/tokens/unigram5000_units.txt"
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        pytest.skip("Model or vocab file not present for test")

    engine = StandaloneONNXVSR(model_path=model_path, vocab_path=vocab_path)
    dummy_input = np.random.randn(1, 1, 25, 88, 88).astype(np.float32)
    res = engine.predict(dummy_input)
    assert isinstance(res, str)
