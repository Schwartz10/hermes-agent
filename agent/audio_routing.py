"""Validation helpers for native audio content parts."""

from __future__ import annotations

import base64
import binascii
import logging
from pathlib import Path
from typing import Any, Dict, Optional

MAX_AUDIO_BYTES = 25 * 1024 * 1024
SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "webm", "flac")

_AUDIO_PART_TYPES = frozenset({"input_audio", "audio"})

_MIME_TO_FORMAT = {
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "mp4",
    "audio/x-m4a": "m4a",
    "audio/mpga": "mpga",
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/webm": "webm",
    # Browser MediaRecorder audio captures commonly arrive as video/webm.
    "video/webm": "webm",
    "audio/flac": "flac",
    "audio/aac": "m4a",
}

logger = logging.getLogger(__name__)


def _format_from_mime(mime_type: str) -> Optional[str]:
    return _MIME_TO_FORMAT.get(str(mime_type or "").split(";", 1)[0].strip().lower())


def _normalize_format(format_value: Any, mime_type: str = "", filename: str = "") -> Optional[str]:
    fmt = str(format_value or "").strip().lower().lstrip(".")
    if fmt in SUPPORTED_AUDIO_FORMATS:
        return fmt
    from_mime = _format_from_mime(mime_type)
    if from_mime:
        return from_mime
    suffix = Path(str(filename or "")).suffix.lower().lstrip(".")
    if suffix in SUPPORTED_AUDIO_FORMATS:
        return suffix
    return None


def normalize_input_audio_part(part: Dict[str, Any], *, validate_data: bool = False) -> Dict[str, Any]:
    """Normalize supported audio shapes to provider-safe OpenAI input_audio."""
    if not isinstance(part, dict):
        raise ValueError("invalid_audio:Audio part must be an object.")

    payload = part.get("input_audio")
    if payload is None:
        payload = part.get("audio")
    if isinstance(payload, str):
        payload = {"data": payload}
    elif not isinstance(payload, dict):
        audio_url = part.get("audio_url") or part.get("url")
        if isinstance(audio_url, str):
            payload = {"data": audio_url}
        else:
            raise ValueError("invalid_audio:Audio parts must include input_audio data.")

    raw_data = payload.get("data") or payload.get("audio_url") or payload.get("url")
    if not isinstance(raw_data, str) or not raw_data.strip():
        raise ValueError("invalid_audio:Audio input must include non-empty base64 data.")

    data_value = raw_data.strip()
    mime_type = str(payload.get("mime_type") or part.get("mime_type") or "").strip().lower()

    if data_value.lower().startswith("data:"):
        header, sep, body = data_value.partition(",")
        if not sep:
            raise ValueError("invalid_audio:Audio data URL must include a comma separator.")
        if ";base64" not in header.lower():
            raise ValueError("invalid_audio:Audio data URL must be base64 encoded.")
        url_mime = header[len("data:"):].split(";", 1)[0].strip().lower()
        if not (url_mime.startswith("audio/") or url_mime == "video/webm"):
            raise ValueError("unsupported_content_type:Only audio data URLs are supported for input_audio.")
        mime_type = mime_type or url_mime
        data_value = body.strip()

    fmt = _normalize_format(
        payload.get("format") or part.get("format"),
        mime_type,
        payload.get("filename") or part.get("filename") or "",
    )
    if not fmt:
        raise ValueError("unsupported_content_type:Audio input must include a supported format.")

    if validate_data:
        try:
            raw = base64.b64decode(data_value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid_audio:Audio input must be valid base64.") from exc
        if not raw:
            raise ValueError("invalid_audio:Audio input is empty.")
        if len(raw) > MAX_AUDIO_BYTES:
            raise ValueError(f"audio_too_large:Audio input exceeds {MAX_AUDIO_BYTES} bytes.")

    return {
        "type": "input_audio",
        "input_audio": {
            "data": data_value,
            "format": fmt,
        },
    }


def content_has_audio_parts(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(part, dict) and str(part.get("type") or "").strip().lower() in _AUDIO_PART_TYPES
        for part in content
    )


def lookup_supports_audio_input(provider: str, model: str, cfg: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Return True/False for known native audio support, None when unknown."""
    if isinstance(cfg, dict):
        from agent.image_routing import _coerce_capability_bool

        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        top = _coerce_capability_bool(model_cfg.get("supports_audio_input"))
        if top is not None:
            return top

        providers_cfg = cfg.get("providers") if isinstance(cfg.get("providers"), dict) else {}
        config_provider = str(model_cfg.get("provider") or "").strip()
        for provider_key in dict.fromkeys(filter(None, (provider, config_provider))):
            provider_cfg = providers_cfg.get(provider_key)
            if not isinstance(provider_cfg, dict):
                continue
            models_cfg = provider_cfg.get("models")
            if not isinstance(models_cfg, dict):
                continue
            per_model = models_cfg.get(model)
            if isinstance(per_model, dict):
                value = _coerce_capability_bool(per_model.get("supports_audio_input"))
                if value is not None:
                    return value

    if not provider or not model:
        return None
    try:
        from agent.models_dev import get_model_capabilities

        caps = get_model_capabilities(provider, model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("audio_routing: models.dev caps lookup failed for %s:%s: %s", provider, model, exc)
        return None
    return None if caps is None else bool(getattr(caps, "supports_audio_input", False))


__all__ = [
    "MAX_AUDIO_BYTES",
    "SUPPORTED_AUDIO_FORMATS",
    "content_has_audio_parts",
    "lookup_supports_audio_input",
    "normalize_input_audio_part",
]
