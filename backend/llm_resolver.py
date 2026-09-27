"""Async Ollama client for homophene correction and language prior resolution."""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert phonetic language reconstructor for a Visual Speech Recognition (VSR / lip-reading) system.
The input text is derived strictly from silent lip kinematics and contains common visual ambiguities:
1. Homophenous substitutions (sounds that look identical on lips):
   - Bilabials: p / b / m (e.g., 'pack' vs 'back' vs 'mack')
   - Alveolars: t / d / n (e.g., 'tip' vs 'dip' vs 'nip')
   - Labiodentals: f / v (e.g., 'fan' vs 'van')
   - Velars: k / g (e.g., 'cane' vs 'gain')
2. Missing unstressed particles or function words (e.g., 'a', 'the', 'is', 'to').
3. The input text is typically in ALL CAPS.

Rules:
- Infer the most probable natural English sentence spoken.
- Do NOT hallucinate or invent new facts, entities, or topics.
- Return ONLY the single corrected English sentence with standard sentence capitalization and punctuation.
- If the input is empty, single-character, or unintelligible noise, return an empty string.
"""


def format_sentence_case(text: str) -> str:
    """Fallback helper to capitalize first letter and format all-caps text."""
    clean = text.strip()
    if not clean:
        return ""
    clean = clean.lower()
    return clean[0].upper() + clean[1:]


class LLMResolver:
    """Resolves homophenous errors and grammatical structure using a local SLM/LLM."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3.5:4b",
        timeout_seconds: float = 6.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_seconds)

    async def is_available(self) -> bool:
        """Check if the local Ollama instance is reachable."""
        try:
            resp = await self.client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def resolve_homophenes(self, raw_vsr_text: str) -> str:
        """Correct raw CTC visual text using Ollama, with graceful fallback to sentence casing."""
        raw_clean = raw_vsr_text.strip()
        if not raw_clean or len(raw_clean) < 2:
            return ""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Input: {raw_clean}"},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 48,
            },
        }

        try:
            response = await self.client.post("/api/chat", json=payload)
            if response.status_code == 200:
                data = response.json()
                content = data.get("message", {}).get("content", "").strip()
                if content:
                    return content
        except Exception as e:
            logger.warning("Ollama homophene resolution failed (%s); falling back to raw formatting.", e)

        return format_sentence_case(raw_clean)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()
