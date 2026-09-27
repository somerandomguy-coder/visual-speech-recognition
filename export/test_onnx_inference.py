"""Pure ONNX Runtime inference test for Auto-AVSR without PyTorch/ESPnet dependencies."""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("onnx_inference")


class StandaloneONNXVSR:
    """Lightweight VSR engine using only ONNX Runtime and a vocabulary file."""

    def __init__(
        self,
        model_path: str = "models/auto_avsr_visual_int8.onnx",
        vocab_path: str = "pipelines/tokens/unigram5000_units.txt",
    ) -> None:
        self.model_path = model_path
        self.vocab_path = vocab_path

        logger.info("Loading ONNX model from %s...", model_path)
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Load vocabulary: 0 is <blank>, followed by units, ending with <eos>
        with open(vocab_path, "r", encoding="utf-8") as f:
            units = [line.strip().split()[0] for line in f if line.strip()]
        self.token_list = ["<blank>"] + units + ["<eos>"]
        logger.info("Vocabulary loaded with %d tokens.", len(self.token_list))

    def ctc_decode(self, logits: np.ndarray) -> str:
        """Standard CTC greedy decoding: argmax -> collapse repeats -> remove blanks -> join."""
        # logits shape: (1, T, VocabSize) or (T, VocabSize)
        if logits.ndim == 3:
            logits = logits[0]

        best_tokens = np.argmax(logits, axis=-1)  # shape (T,)

        # 1. Collapse consecutive duplicates
        collapsed = []
        prev = None
        for tok in best_tokens:
            if tok != prev:
                collapsed.append(tok)
                prev = tok

        # 2. Remove blanks (index 0) and <eos> (last index)
        eos_id = len(self.token_list) - 1
        filtered = [t for t in collapsed if t != 0 and t != eos_id]

        # 3. Map to tokens and format
        words = []
        for t in filtered:
            if t < len(self.token_list):
                words.append(self.token_list[t])

        text = "".join(words)
        # Clean SentencePiece prefix symbols ' ' or '▁'
        text = text.replace(" ", " ").replace("▁", " ").strip()
        return text

    def predict(self, normalized_frames_88x88: np.ndarray) -> str:
        """Run inference on normalized 88x88 frames (1, 1, T, 88, 88)."""
        if normalized_frames_88x88.ndim == 4:
            # (1, T, 88, 88) -> (1, 1, T, 88, 88)
            tensor = np.expand_dims(normalized_frames_88x88, axis=1)
        elif normalized_frames_88x88.ndim == 3:
            # (T, 88, 88) -> (1, 1, T, 88, 88)
            tensor = normalized_frames_88x88.reshape(1, 1, -1, 88, 88)
        else:
            tensor = normalized_frames_88x88

        t0 = time.perf_counter()
        ort_outputs = self.session.run([self.output_name], {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logits = ort_outputs[0]
        decoded = self.ctc_decode(logits)
        logger.info("ONNX Inference took %.2f ms | Result: '%s'", elapsed_ms, decoded)
        return decoded


def main():
    parser = argparse.ArgumentParser(description="Test Standalone ONNX VSR inference")
    parser.add_argument("--model", default="models/auto_avsr_visual_int8.onnx", help="Path to ONNX model")
    args = parser.parse_args()

    engine = StandaloneONNXVSR(model_path=args.model)

    # Test with synthetic input of 37 frames
    dummy_input = np.random.randn(1, 1, 37, 88, 88).astype(np.float32)
    result = engine.predict(dummy_input)
    print(f"\nSynthetic Test Succeeded! Result: '{result}'")


if __name__ == "__main__":
    main()
