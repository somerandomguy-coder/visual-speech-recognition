"""Auto-AVSR visual speech recognition engine and sliding-window token merger."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


def merge_sliding_tokens(prev_text: str, new_chunk: str) -> Tuple[str, str]:
    """Merge overlapping text chunks from sliding-window inferences.

    Returns:
        (merged_full_text, committed_stable_text)
    """
    prev_clean = prev_text.strip()
    new_clean = new_chunk.strip()

    if not prev_clean:
        return new_clean, ""
    if not new_clean:
        return prev_clean, prev_clean

    prev_words = prev_clean.split()
    new_words = new_clean.split()

    # Look for matching suffix of prev_words and prefix of new_words
    max_k = min(len(prev_words), len(new_words))
    overlap_k = 0

    for k in range(max_k, 0, -1):
        if [w.upper() for w in prev_words[-k:]] == [w.upper() for w in new_words[:k]]:
            overlap_k = k
            break

    if overlap_k > 0:
        merged_words = prev_words + new_words[overlap_k:]
    else:
        # Check partial word overlap at boundary (e.g. "WHA" vs "WHAT")
        last_prev = prev_words[-1].upper()
        first_new = new_words[0].upper()
        if first_new.startswith(last_prev) or last_prev.startswith(first_new):
            # Replace partial with the longer word
            chosen = new_words[0] if len(new_words[0]) >= len(prev_words[-1]) else prev_words[-1]
            merged_words = prev_words[:-1] + [chosen] + new_words[1:]
        else:
            merged_words = prev_words + new_words

    merged_text = " ".join(merged_words)
    # The last 1-2 words are still inside the active sliding window and subject to change
    committed_count = max(0, len(merged_words) - 1)
    committed_text = " ".join(merged_words[:committed_count])

    return merged_text, committed_text


class VSREngine:
    """Headless Auto-AVSR inference engine with 1.5s sliding-window buffer."""

    def __init__(
        self,
        config_path: str = "backend/configs/LRS3_V_WER19.1.ini",
        device: Optional[str] = None,
        window_frames: int = 37,  # 1.5s at 25 FPS
        step_frames: int = 12,    # 0.5s step
    ) -> None:
        self.config_path = config_path
        self.window_frames = window_frames
        self.step_frames = step_frames

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.pipeline = None
        self.frame_buffer: List[np.ndarray] = []
        self.full_transcript: str = ""
        self.committed_transcript: str = ""

    def load_model(self) -> None:
        """Load Auto-AVSR model pipeline into memory/GPU."""
        from pipelines.pipeline import InferencePipeline

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        logger.info("Loading Auto-AVSR visual model on %s...", self.device)
        self.pipeline = InferencePipeline(
            config_filename=self.config_path,
            detector="mediapipe",
            face_track=False,
            device=self.device,
        )
        logger.info("Auto-AVSR visual model loaded successfully.")

    def reset(self) -> None:
        """Reset internal frame buffer and accumulated transcripts."""
        self.frame_buffer.clear()
        self.full_transcript = ""
        self.committed_transcript = ""

    def predict_frames(self, frames_96x96: List[np.ndarray]) -> str:
        """Run Auto-AVSR model inference directly on a sequence of 96x96 grayscale mouth frames."""
        if not frames_96x96:
            return ""

        if self.pipeline is None:
            self.load_model()

        # Stack into numpy array (T, 96, 96)
        raw_arr = np.stack(frames_96x96, axis=0)
        raw_tensor = torch.from_numpy(raw_arr).float()

        if hasattr(self.pipeline, "dataloader") and hasattr(self.pipeline.dataloader, "video_transform"):
            transformed = self.pipeline.dataloader.video_transform(raw_tensor)
        else:
            # Fallback transform if dataloader is not present (or in mock)
            t = raw_tensor.unsqueeze(0)  # (1, T, 96, 96)
            if t.shape[-2] >= 88 and t.shape[-1] >= 88:
                h_start = (t.shape[-2] - 88) // 2
                w_start = (t.shape[-1] - 88) // 2
                transformed = t[:, :, h_start : h_start + 88, w_start : w_start + 88] / 255.0
            else:
                transformed = t / 255.0
            transformed = (transformed - 0.421) / 0.165

        try:
            with torch.no_grad():
                transcript = self.pipeline.model.infer(transformed.to(self.device))
                return transcript.strip()
        except Exception as e:
            logger.error("Auto-AVSR inference error: %s", e)
            return ""

    def process_sliding_window(self, new_frames_25fps: List[np.ndarray]) -> dict:
        """Ingest new 25 FPS mouth frames, evaluate the 1.5s sliding window when ready,

        and return current transcripts.
        """
        self.frame_buffer.extend(new_frames_25fps)

        raw_chunk = ""
        # When we have at least window_frames (37 frames = 1.5s), run inference
        if len(self.frame_buffer) >= self.window_frames:
            eval_frames = self.frame_buffer[: self.window_frames]
            raw_chunk = self.predict_frames(eval_frames)

            # Slide window forward by step_frames
            self.frame_buffer = self.frame_buffer[self.step_frames :]

            if raw_chunk:
                self.full_transcript, self.committed_transcript = merge_sliding_tokens(
                    self.full_transcript, raw_chunk
                )

        return {
            "raw_chunk": raw_chunk,
            "full_transcript": self.full_transcript,
            "committed_transcript": self.committed_transcript,
        }
