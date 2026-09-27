"""Export Auto-AVSR visual Conformer model to standalone ONNX format."""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from backend.vsr_engine import VSREngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_onnx")


class VisualSpeechEncoder(nn.Module):
    """Clean wrapper isolating visual 3D frontend + Conformer encoder + CTC projection."""

    def __init__(self, encoder: nn.Module, ctc_lo: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.ctc_lo = ctc_lo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: (B, 1, T, 88, 88) float32 normalized mouth video tensor
        :return: (B, T, VocabSize) CTC classification logits
        """
        enc_out, _ = self.encoder(x, None)
        logits = self.ctc_lo(enc_out)
        return logits


def export_vsr_to_onnx(
    output_path: str = "models/auto_avsr_visual.onnx",
    config_path: str = "backend/configs/LRS3_V_WER19.1.ini",
    opset_version: int = 17,
) -> Path:
    """Load PyTorch checkpoint, wrap visual Conformer, and export to ONNX."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing VSREngine on CPU...")
    engine = VSREngine(config_path=config_path, device="cpu")
    engine.load_model()

    model = engine.pipeline.model.model
    model.eval()

    encoder_wrapper = VisualSpeechEncoder(model.encoder, model.ctc.ctc_lo)
    encoder_wrapper.eval()

    # Dummy tensor for tracing: (B=1, C=1, T=37, H=88, W=88) ~ 1.5s at 25 FPS
    dummy_input = torch.randn(1, 1, 37, 88, 88, dtype=torch.float32)

    logger.info("Testing PyTorch forward pass...")
    with torch.no_grad():
        pt_out = encoder_wrapper(dummy_input)
    logger.info("PyTorch output shape: %s", pt_out.shape)

    logger.info("Exporting to ONNX at %s (opset=%d)...", output_file, opset_version)
    torch.onnx.export(
        encoder_wrapper,
        dummy_input,
        str(output_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["video_frames"],
        output_names=["logits"],
        dynamic_axes={
            "video_frames": {0: "batch", 2: "time"},
            "logits": {0: "batch", 1: "time"},
        },
    )

    logger.info("Verifying ONNX model structure...")
    onnx_model = onnx.load(str(output_file))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model check passed successfully!")

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    logger.info("Exported ONNX file size: %.2f MB", file_size_mb)

    # Verify ONNX Runtime parity
    logger.info("Verifying numerical parity via ONNX Runtime...")
    session = ort.InferenceSession(str(output_file), providers=["CPUExecutionProvider"])
    ort_inputs = {"video_frames": dummy_input.numpy()}
    ort_out = session.run(["logits"], ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out.numpy() - ort_out))
    logger.info("Maximum absolute difference between PyTorch and ONNX Runtime: %.6f", max_diff)
    assert max_diff < 1e-3, f"Parity check failed: max_diff = {max_diff}"
    logger.info("Numerical parity check PASSED (diff < 1e-3).")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Auto-AVSR visual model to ONNX")
    parser.add_argument("--output", default="models/auto_avsr_visual.onnx", help="Output ONNX path")
    parser.add_argument("--config", default="backend/configs/LRS3_V_WER19.1.ini", help="Model config")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_vsr_to_onnx(output_path=args.output, config_path=args.config, opset_version=args.opset)
