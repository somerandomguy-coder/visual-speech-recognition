"""Quantize the exported Auto-AVSR ONNX model to INT8 and FP16 for mobile deployment."""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quantize")


def quantize_int8(
    input_model_path: str = "models/auto_avsr_visual.onnx",
    output_model_path: str = "models/auto_avsr_visual_int8.onnx",
) -> Path:
    """Apply dynamic INT8 quantization to Linear and MatMul layers."""
    input_path = Path(input_model_path)
    output_path = Path(output_model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found at {input_path}")

    logger.info("Starting dynamic INT8 quantization on %s...", input_path)
    t0 = time.perf_counter()

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Linear"],
        per_channel=True,
        reduce_range=False,
    )

    elapsed = time.perf_counter() - t0
    original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    quantized_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    logger.info("INT8 quantization complete in %.2f seconds.", elapsed)
    logger.info("Original model size:   %.2f MB", original_size_mb)
    logger.info("Quantized model size:  %.2f MB (%.1f%% reduction)",
                quantized_size_mb, (1.0 - quantized_size_mb / original_size_mb) * 100)

    # Benchmark latency comparison on CPU
    logger.info("Comparing inference latency (FP32 vs INT8)...")
    dummy_input = np.random.randn(1, 1, 37, 88, 88).astype(np.float32)

    # FP32 Session
    sess_fp32 = ort.InferenceSession(str(input_path), providers=["CPUExecutionProvider"])
    # Warmup
    sess_fp32.run(["logits"], {"video_frames": dummy_input})
    t_fp32 = time.perf_counter()
    for _ in range(5):
        sess_fp32.run(["logits"], {"video_frames": dummy_input})
    avg_fp32_ms = (time.perf_counter() - t_fp32) / 5 * 1000.0

    # INT8 Session
    sess_int8 = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    # Warmup
    sess_int8.run(["logits"], {"video_frames": dummy_input})
    t_int8 = time.perf_counter()
    for _ in range(5):
        sess_int8.run(["logits"], {"video_frames": dummy_input})
    avg_int8_ms = (time.perf_counter() - t_int8) / 5 * 1000.0

    logger.info("FP32 Latency: %.2f ms", avg_fp32_ms)
    logger.info("INT8 Latency: %.2f ms (Speedup: %.2fx)", avg_int8_ms, avg_fp32_ms / avg_int8_ms if avg_int8_ms > 0 else 1.0)

    return output_path


def convert_fp16(
    input_model_path: str = "models/auto_avsr_visual.onnx",
    output_model_path: str = "models/auto_avsr_visual_fp16.onnx",
) -> Path:
    """Convert FP32 model to FP16 for mobile GPU/Metal/Vulkan execution."""
    input_path = Path(input_model_path)
    output_path = Path(output_model_path)

    try:
        from onnxconverter_common import float16
        logger.info("Converting %s to FP16...", input_path)
        model = onnx.load(str(input_path))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
        onnx.save(model_fp16, str(output_path))
        fp16_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info("FP16 model exported successfully: %.2f MB", fp16_size_mb)
        return output_path
    except ImportError:
        logger.warning("onnxconverter_common not installed. Skipping FP16 conversion.")
        return input_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Auto-AVSR ONNX model")
    parser.add_argument("--input", default="models/auto_avsr_visual.onnx", help="Input FP32 ONNX model")
    parser.add_argument("--output_int8", default="models/auto_avsr_visual_int8.onnx", help="Output INT8 ONNX path")
    parser.add_argument("--output_fp16", default="models/auto_avsr_visual_fp16.onnx", help="Output FP16 ONNX path")
    args = parser.parse_args()

    quantize_int8(args.input, args.output_int8)
    convert_fp16(args.input, args.output_fp16)
