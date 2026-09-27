"""Benchmark the completely offline Visual Speech Recognition pipeline (ONNX VSR + GGUF SLM)."""

import logging
import os
import time
from pathlib import Path
import numpy as np

from export.test_onnx_inference import StandaloneONNXVSR
from backend.offline_slm import OfflineSLMEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark_offline")

ONNX_MODEL_PATH = "models/auto_avsr_visual_int8.onnx"
GGUF_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
VOCAB_PATH = "pipelines/tokens/unigram5000_units.txt"


def benchmark_offline():
    print("=" * 65)
    print(" 100% OFFLINE ZERO-NETWORK VISUAL SPEECH RECOGNITION BENCHMARK")
    print("=" * 65)

    if not os.path.exists(ONNX_MODEL_PATH):
        raise FileNotFoundError(f"Missing ONNX model: {ONNX_MODEL_PATH}")
    if not os.path.exists(GGUF_MODEL_PATH):
        raise FileNotFoundError(f"Missing GGUF model: {GGUF_MODEL_PATH}")

    # 1. Load ONNX VSR Engine
    t0 = time.perf_counter()
    vsr_engine = StandaloneONNXVSR(model_path=ONNX_MODEL_PATH, vocab_path=VOCAB_PATH)
    vsr_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[1] Standalone ONNX VSR loaded in {vsr_load_ms:.1f} ms")

    # 2. Load Offline GGUF SLM
    t0 = time.perf_counter()
    slm_engine = OfflineSLMEngine(model_path=GGUF_MODEL_PATH, n_ctx=512, n_threads=4)
    slm_load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[2] Offline GGUF SLM loaded in {slm_load_ms:.1f} ms")

    # 3. Benchmark VSR Inference Latency
    print("\n--- Benchmarking ONNX VSR (1.5s sliding window: 37 frames) ---")
    dummy_window = np.random.randn(1, 1, 37, 88, 88).astype(np.float32)

    # Warmup
    vsr_engine.predict(dummy_window)

    vsr_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        vsr_engine.predict(dummy_window)
        vsr_times.append((time.perf_counter() - t0) * 1000.0)

    avg_vsr_ms = float(np.mean(vsr_times))
    print(f"  Average ONNX VSR latency: {avg_vsr_ms:.2f} ms (~{1500.0 / avg_vsr_ms:.1f}x real-time)")

    # 4. Benchmark SLM Homophene Resolution Latency
    print("\n--- Benchmarking Offline GGUF SLM Homophene Resolution ---")
    test_phrases = [
        "HELLO HOW ARE EOU",
        "WE ARE COIN TO THE SHOB",
        "I WANT TO CO TO THE BARK TO PLAY PALL",
        "PLEASE TURN ON THE LIGHTS IN THE ROON",
    ]

    # Warmup
    slm_engine.resolve(test_phrases[0])

    slm_times = []
    print(f"{'Input Visual CTC Text':<40} | {'Corrected Output':<35} | {'Latency'}")
    print("-" * 90)
    for phrase in test_phrases:
        t0 = time.perf_counter()
        corrected = slm_engine.resolve(phrase)
        lat = (time.perf_counter() - t0) * 1000.0
        slm_times.append(lat)
        print(f"{phrase:<40} | {corrected:<35} | {lat:.1f} ms")

    avg_slm_ms = float(np.mean(slm_times))
    print(f"\n  Average SLM resolution latency: {avg_slm_ms:.2f} ms")

    total_pipeline_ms = avg_vsr_ms + avg_slm_ms
    print("\n" + "=" * 65)
    print(f" TOTAL OFFLINE PIPELINE LATENCY: {total_pipeline_ms:.2f} ms")
    print(f"   - ONNX Visual Speech Conformer: {avg_vsr_ms:.2f} ms")
    print(f"   - llama.cpp GGUF Resolution:   {avg_slm_ms:.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    benchmark_offline()
