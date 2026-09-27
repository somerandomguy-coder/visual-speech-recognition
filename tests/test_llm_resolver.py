"""Unit tests for LLMResolver."""

from unittest.mock import AsyncMock, patch
import httpx
import pytest
from backend.llm_resolver import LLMResolver, format_sentence_case


def test_format_sentence_case():
    assert format_sentence_case("HELLO WORLD") == "Hello world"
    assert format_sentence_case("") == ""
    assert format_sentence_case("a") == "A"


@pytest.mark.asyncio
async def test_homophene_resolution_success():
    resolver = LLMResolver()
    mock_response = httpx.Response(
        status_code=200,
        json={"message": {"content": "I want to go to the park."}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(resolver.client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await resolver.resolve_homophenes("I WANT TO CO TO THE BARK")
        assert result == "I want to go to the park."

    await resolver.close()


@pytest.mark.asyncio
async def test_homophene_resolution_fallback_on_error():
    resolver = LLMResolver()

    with patch.object(resolver.client, "post", side_effect=httpx.ConnectError("Connection refused")):
        # On connection failure, must fallback gracefully to sentence case
        result = await resolver.resolve_homophenes("GOOD MORNING EVERYONE")
        assert result == "Good morning everyone"

    await resolver.close()
