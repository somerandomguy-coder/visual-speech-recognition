"""Offline SLM engine using llama.cpp to resolve visual homophenes on-device."""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from backend.llm_resolver import SYSTEM_PROMPT, format_sentence_case

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("offline_slm")


class OfflineSLMEngine:
    """Runs a local quantized GGUF small language model completely offline using llama.cpp."""

    def __init__(
        self,
        model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        n_ctx: int = 512,
        n_threads: int = 4,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self.llm = None

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Offline SLM GGUF model not found at: {model_path}")

        self._load_model()

    def _load_model(self) -> None:
        from llama_cpp import Llama

        logger.info("Loading offline GGUF SLM from %s (n_ctx=%d, threads=%d)...",
                    self.model_path, self.n_ctx, self.n_threads)
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=self.verbose,
        )
        logger.info("Offline SLM loaded successfully.")

    def resolve(self, raw_vsr_text: str) -> str:
        """Resolve visual speech ambiguities and homophenes completely offline."""
        raw_clean = raw_vsr_text.strip()
        if not raw_clean or len(raw_clean) < 2:
            return ""

        if self.llm is None:
            return format_sentence_case(raw_clean)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert visual speech corrector for silent lip-reading CTC output. "
                    "Fix phonetic homophene substitutions (b/p/m, d/t/n, g/k) and punctuation. "
                    "Output ONLY the corrected English sentence."
                ),
            },
            {"role": "user", "content": "HELLO HOW ARE EOU"},
            {"role": "assistant", "content": "Hello, how are you?"},
            {"role": "user", "content": "WE ARE COIN TO THE SHOB"},
            {"role": "assistant", "content": "We are going to the shop."},
            {"role": "user", "content": raw_clean},
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.1,
                top_p=0.9,
                max_tokens=48,
            )
            content = response["choices"][0]["message"]["content"].strip()
            # Remove any residual quotation marks or leading prefixes
            content = re.sub(r'^["\']|["\']$', "", content).strip()
            if content:
                return content
        except Exception as e:
            logger.warning("Offline SLM inference exception: %s", e)

        return format_sentence_case(raw_clean)


if __name__ == "__main__":
    engine = OfflineSLMEngine()
    sample = "I WANT TO CO TO THE BARK TO PLAY PALL"
    print(f"Input:     {sample}")
    print(f"Corrected: {engine.resolve(sample)}")
