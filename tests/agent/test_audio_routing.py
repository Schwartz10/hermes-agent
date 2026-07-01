from unittest.mock import patch

import pytest

from agent.audio_routing import (
    content_has_audio_parts,
    lookup_supports_audio_input,
    normalize_input_audio_part,
)


def test_normalize_input_audio_part_accepts_openai_shape():
    out = normalize_input_audio_part(
        {
            "type": "input_audio",
            "input_audio": {
                "data": "ZmFrZQ==",
                "format": "ogg",
                "mime_type": "audio/ogg",
                "filename": "voice.ogg",
            },
        },
        validate_data=True,
    )

    assert out == {
        "type": "input_audio",
        "input_audio": {"data": "ZmFrZQ==", "format": "ogg"},
    }


def test_normalize_input_audio_part_accepts_data_url():
    out = normalize_input_audio_part(
        {"type": "input_audio", "audio_url": "data:audio/wav;base64,ZmFrZQ=="},
        validate_data=True,
    )

    assert out["input_audio"]["data"] == "ZmFrZQ=="
    assert out["input_audio"]["format"] == "wav"


def test_normalize_input_audio_part_rejects_bad_base64():
    with pytest.raises(ValueError, match="invalid_audio:"):
        normalize_input_audio_part(
            {"type": "input_audio", "input_audio": {"data": "not base64", "format": "ogg"}},
            validate_data=True,
        )


def test_normalize_input_audio_part_rejects_missing_format():
    with pytest.raises(ValueError, match="unsupported_content_type:"):
        normalize_input_audio_part({"type": "input_audio", "input_audio": {"data": "ZmFrZQ=="}})


def test_content_has_audio_parts():
    assert content_has_audio_parts([
        {"type": "text", "text": "listen"},
        {"type": "input_audio", "input_audio": {"data": "ZmFrZQ==", "format": "ogg"}},
    ])
    assert not content_has_audio_parts([{"type": "text", "text": "listen"}])


def test_lookup_supports_audio_input_top_level_override():
    cfg = {"model": {"supports_audio_input": True}}
    with patch("agent.models_dev.get_model_capabilities", return_value=None):
        assert lookup_supports_audio_input("custom", "voice-model", cfg) is True


def test_lookup_supports_audio_input_provider_model_override():
    cfg = {
        "providers": {
            "custom": {"models": {"voice-model": {"supports_audio_input": "false"}}},
        },
    }
    assert lookup_supports_audio_input("custom", "voice-model", cfg) is False


def test_lookup_supports_audio_input_models_dev_fallback():
    fake_caps = type("Caps", (), {"supports_audio_input": True})()
    with patch("agent.models_dev.get_model_capabilities", return_value=fake_caps):
        assert lookup_supports_audio_input("openai", "gpt-audio", {}) is True


def test_lookup_supports_audio_input_unknown_returns_none():
    with patch("agent.models_dev.get_model_capabilities", return_value=None):
        assert lookup_supports_audio_input("custom", "unknown", {}) is None
