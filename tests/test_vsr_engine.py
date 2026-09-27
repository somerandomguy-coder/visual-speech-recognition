"""Unit tests for VSREngine and sliding token merger."""

import pytest
from backend.vsr_engine import merge_sliding_tokens


def test_merge_sliding_tokens_exact_overlap():
    prev_text = "HELLO HOW"
    new_chunk = "HOW ARE YOU"
    merged, committed = merge_sliding_tokens(prev_text, new_chunk)
    assert merged == "HELLO HOW ARE YOU"
    assert "HELLO" in committed


def test_merge_sliding_tokens_multiple_word_overlap():
    prev_text = "WELCOME TO THE FUTURE"
    new_chunk = "THE FUTURE OF SPEECH"
    merged, committed = merge_sliding_tokens(prev_text, new_chunk)
    assert merged == "WELCOME TO THE FUTURE OF SPEECH"
    assert "WELCOME TO THE" in committed


def test_merge_sliding_tokens_partial_word_prefix():
    prev_text = "WE ARE GOIN"
    new_chunk = "GOING TO THE STORE"
    merged, committed = merge_sliding_tokens(prev_text, new_chunk)
    assert merged == "WE ARE GOING TO THE STORE"


def test_merge_sliding_tokens_no_overlap():
    prev_text = "HELLO"
    new_chunk = "WORLD"
    merged, committed = merge_sliding_tokens(prev_text, new_chunk)
    assert merged == "HELLO WORLD"
