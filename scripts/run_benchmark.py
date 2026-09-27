"""Benchmark pipeline components: FaceMesh, Auto-AVSR inference, and Ollama resolution."""

import asyncio
import time
import numpy as np
import torch
from backend.face_tracker import FaceTracker
from backend.llm_resolver import LLMResolver
from backend.resampler import TemporalResampler
from backend.vsr_engine import VSREngine


def benchmark_vision():
    print("\n--- 1. Vision Preprocessing Benchmark ---")
    tracker = FaceTracker()
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # Warmup
    tracker.process_frame(dummy_frame)

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        tracker.process_frame(dummy_frame)
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = np.mean(times)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    print(f"FaceTracker latency: {avg_ms:.2f} ms ({fps:.1f} FPS equivalent)")


def benchmark_vsr_engine():
    print("\n--- 2. Auto-AVSR Inference Benchmark ---")
    device = "cpu"
    print(f"Target device: {device} (Auto-AVSR visual Conformer on CPU)")

    engine = VSREngine(device=device)
    try:
        engine.load_model()
    except Exception as e:
        print(f"Could not load Auto-AVSR model for benchmark: {e}")
        return

    # Simulate 37 frames (1.5s window at 25 FPS)
    dummy_window = [np.full((96, 96), 128, dtype=np.uint8) for _ in range(37)]

    # Warmup
    engine.predict_frames(dummy_window)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        engine.predict_frames(dummy_window)
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = np.mean(times)
    print(f"Auto-AVSR 1.5s window inference latency: {avg_ms:.2f} ms")


async def benchmark_llm():
    print("\n--- 3. Ollama Homophene Resolver Benchmark ---")
    resolver = LLMResolver()
    available = await resolver.is_available()
    print(f"Ollama server available: {available}")
    if not available:
        await resolver.close()
        return

    sample_input = "I WANT TO CO TO THE BARK TO PLAY PALL"
    t0 = time.perf_counter()
    corrected = await resolver.resolve_homophenes(sample_input)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Input:     {sample_input}")
    print(f"Corrected: {corrected}")
    print(f"Resolution latency: {latency_ms:.2f} ms")
    await resolver.close()


def main():
    benchmark_vision()
    benchmark_vsr_engine()
    asyncio.run(benchmark_llm())
    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
