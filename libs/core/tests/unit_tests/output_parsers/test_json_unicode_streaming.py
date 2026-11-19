"""Regression tests for Unicode escapes in incrementally streamed JSON.

Examples model travel, commerce, payments, customer support, and other
real tool arguments rather than placeholder-only ASCII values. JSON
RFC 8259 permits \uXXXX escapes and surrogate pairs for non-BMP characters.
"""

import json
import re

import pytest

from langchain_core.messages import AIMessageChunk
from langchain_core.utils.json import parse_partial_json


@pytest.mark.parametrize(
    ("domain", "value"),
    [
        ("travel_city", "München"),
        ("travel_destination", "São Paulo"),
        ("geocoding_japanese", "東京都"),
        ("customer_name", "Zoë François"),
        ("restaurant_search", "Crème brûlée"),
        ("payment_currency", "₹1,250"),
        ("customer_support_arabic", "مرحبا بالعالم"),
        ("healthcare_japanese", "患者情報"),
        ("shipping_greek", "Αθήνα"),
        ("messaging_emoji", "Delivery confirmed 🚀"),
        ("customer_name_combining_mark", "Jose\u0301"),
    ],
)
def test_streamed_unicode_escape_preserves_key(
    domain: str, value: str
) -> None:
    """Every cut inside an escape retains previously streamed value data."""
    encoded = json.dumps({"value": value}, ensure_ascii=True)
    assert "\\u" in encoded, domain
    for match in re.finditer(r"\\u[0-9a-fA-F]{4}", encoded):
        for digits in range(4):
            cut = match.start() + 2 + digits
            partial = encoded[:cut]
            # The completed part of the value can be decoded independently.
            expected = json.loads(encoded[: match.start()] + '"}')
            assert parse_partial_json(partial) == expected, (
                f"{domain}: truncated after {digits} hex digits: {partial!r}"
            )
    assert parse_partial_json(encoded) == {"value": value}


@pytest.mark.parametrize(
    ("partial", "expected"),
    [
        (r'{"city": "M\u', {"city": "M"}),
        (r'{"city": "M\u0', {"city": "M"}),
        (r'{"city": "M\u00', {"city": "M"}),
        (r'{"city": "M\u00f', {"city": "M"}),
        (r'{"flight": {"arrival_city": "Z\u00', {"flight": {"arrival_city": "Z"}}),
        (r'{"cities": ["Paris", "M\u00', {"cities": ["Paris", "M"]}),
        (r'{"pattern": "literal \\u00', {"pattern": r"literal \u00"}),
    ],
)
def test_streamed_unicode_nested_and_escape_cases(
    partial: str, expected: dict
) -> None:
    assert parse_partial_json(partial) == expected


def test_streamed_tool_call_preserves_city_during_unicode_escape() -> None:
    """The user-visible tool-call arguments must not disappear mid-stream."""
    for args, expected in [
        (r'{"city": "M', {"city": "M"}),
        (r'{"city": "M\u00', {"city": "M"}),
        (r'{"city": "M\u00fc', {"city": "Mü"}),
    ]:
        chunk = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"name": "weather", "args": args, "id": "call-1", "index": 0}
            ],
        )
        assert chunk.tool_calls[0]["args"] == expected


def test_complete_surrogate_pair_in_emoji_tool_argument() -> None:
    """A full UTF-16 surrogate pair must decode to the original emoji."""
    encoded = json.dumps({"status": "Ready 🚀"}, ensure_ascii=True)
    assert r"\ud83d\ude80" in encoded
    assert parse_partial_json(encoded) == {"status": "Ready 🚀"}
