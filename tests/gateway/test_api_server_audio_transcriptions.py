"""Tests for the API-server STT utility endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_mod
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli.auth import AuthError


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", adapter._handle_audio_transcriptions)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _multipart(
    *,
    model: str | None = "whisper-1",
    audio: bytes | None = b"fake-audio",
    filename: str = "voice.ogg",
    content_type: str = "audio/ogg",
    file_field: str = "file",
) -> FormData:
    form = FormData()
    if model is not None:
        form.add_field("model", model)
    if audio is not None:
        form.add_field(file_field, audio, filename=filename, content_type=content_type)
    return form


@pytest.mark.asyncio
async def test_transcription_succeeds_with_codex_auth():
    adapter = _make_adapter()
    captured = {}

    async def fake_post(**kwargs):
        captured.update(kwargs)
        return "hello world", None

    app = _create_app(adapter)
    with patch("hermes_cli.runtime_provider.resolve_requested_provider", return_value="openai-codex"), \
         patch(
             "hermes_cli.auth.resolve_codex_runtime_credentials",
             return_value={
                 "provider": "openai-codex",
                 "base_url": "https://chatgpt.example/backend-api/codex",
                 "api_key": "codex-token",
             },
         ), \
         patch.object(adapter, "_post_audio_transcription", side_effect=fake_post), \
         patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart(model="gpt-4o-transcribe"))
            body = await resp.json()

    assert resp.status == 200
    assert body == {
        "text": "hello world",
        "model": "gpt-4o-transcribe",
        "provider": "openai-codex",
    }
    assert captured["runtime"]["provider"] == "openai-codex"
    assert captured["runtime"]["base_url"] == "https://chatgpt.example/backend-api/codex"
    assert captured["runtime"]["api_key"] == "codex-token"
    assert captured["model"] == "gpt-4o-transcribe"
    assert captured["data"] == b"fake-audio"
    mock_run.assert_not_called()
    assert adapter._session_db is None


@pytest.mark.asyncio
async def test_transcription_succeeds_with_custom_runtime_proxy():
    adapter = _make_adapter()
    captured = {}

    async def fake_post(**kwargs):
        captured.update(kwargs)
        return "custom transcript", None

    app = _create_app(adapter)
    with (
        patch("hermes_cli.runtime_provider.resolve_requested_provider", return_value="custom"),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "custom",
                "base_url": "http://william-proxy.local/v1",
                "api_key": "proxy-key",
            },
        ),
        patch.object(adapter, "_post_audio_transcription", side_effect=fake_post),
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart(model="william-stt"))
            body = await resp.json()

    assert resp.status == 200
    assert body == {"text": "custom transcript", "model": "william-stt", "provider": "custom"}
    assert captured["runtime"] == {
        "provider": "custom",
        "base_url": "http://william-proxy.local/v1",
        "api_key": "proxy-key",
    }


@pytest.mark.asyncio
async def test_post_audio_transcription_forwards_openai_compatible_multipart():
    adapter = _make_adapter()
    captured = {}

    async def upstream_handler(request):
        captured["path"] = request.path
        captured["authorization"] = request.headers.get("Authorization")
        reader = await request.multipart()
        fields = {}
        async for part in reader:
            if part.name == "model":
                fields["model"] = await part.text()
            elif part.name == "file":
                fields["filename"] = part.filename
                fields["content_type"] = part.headers.get("Content-Type")
                fields["file"] = await part.read()
        captured["fields"] = fields
        return web.json_response({"text": "from upstream"})

    upstream = web.Application()
    upstream.router.add_post("/v1/audio/transcriptions", upstream_handler)
    server = TestServer(upstream)
    await server.start_server()
    try:
        transcript, err = await adapter._post_audio_transcription(
            runtime={
                "provider": "custom",
                "base_url": str(server.make_url("/v1")).rstrip("/"),
                "api_key": "proxy-key",
            },
            model="william-stt",
            filename="voice.ogg",
            content_type="audio/ogg",
            data=b"fake-audio",
        )
    finally:
        await server.close()

    assert err is None
    assert transcript == "from upstream"
    assert captured["path"] == "/v1/audio/transcriptions"
    assert captured["authorization"] == "Bearer proxy-key"
    assert captured["fields"] == {
        "model": "william-stt",
        "filename": "voice.ogg",
        "content_type": "audio/ogg",
        "file": b"fake-audio",
    }


@pytest.mark.asyncio
async def test_missing_codex_auth_returns_auth_error():
    adapter = _make_adapter()
    app = _create_app(adapter)
    with patch("hermes_cli.runtime_provider.resolve_requested_provider", return_value="openai-codex"), \
         patch(
             "hermes_cli.auth.resolve_codex_runtime_credentials",
             side_effect=AuthError(
                 "No Codex credentials stored. Run `hermes auth` to authenticate.",
                 provider="openai-codex",
                 code="codex_auth_missing",
                 relogin_required=True,
             ),
         ), \
         patch.object(adapter, "_post_audio_transcription", new_callable=AsyncMock) as mock_post:
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart())
            body = await resp.json()

    assert resp.status == 401
    assert body["error"]["code"] == "codex_auth_missing"
    assert "No Codex credentials" in body["error"]["message"]
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_codex_auth_error_does_not_use_runtime_fallback():
    adapter = _make_adapter()
    app = _create_app(adapter)
    with patch("hermes_cli.runtime_provider.resolve_requested_provider", return_value="openai-codex"), \
         patch(
             "hermes_cli.auth.resolve_codex_runtime_credentials",
             side_effect=AuthError(
                 "No Codex credentials stored.",
                 provider="openai-codex",
                 code="codex_auth_missing",
                 relogin_required=True,
             ),
         ), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_runtime, \
         patch.object(adapter, "_post_audio_transcription", new_callable=AsyncMock) as mock_post:
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart())
            body = await resp.json()

    assert resp.status == 401
    assert body["error"]["code"] == "codex_auth_missing"
    mock_runtime.assert_not_called()
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_auto_codex_auth_error_does_not_use_runtime_fallback():
    adapter = _make_adapter()
    app = _create_app(adapter)
    auth_error = AuthError(
        "No Codex credentials stored.",
        provider="openai-codex",
        code="codex_auth_missing",
        relogin_required=True,
    )
    with (
        patch("hermes_cli.runtime_provider.resolve_requested_provider", return_value="auto"),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=auth_error) as mock_runtime,
        patch.object(adapter, "_post_audio_transcription", new_callable=AsyncMock) as mock_post,
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart())
            body = await resp.json()

    assert resp.status == 401
    assert body["error"]["code"] == "codex_auth_missing"
    mock_runtime.assert_called_once_with(requested="auto", allow_auto_codex_fallback=False)
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_missing_custom_runtime_credentials_returns_config_error():
    adapter = _make_adapter()
    app = _create_app(adapter)
    with (
        patch("hermes_cli.runtime_provider.resolve_requested_provider", return_value="custom"),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"provider": "custom", "base_url": "", "api_key": ""},
        ),
        patch.object(adapter, "_post_audio_transcription", new_callable=AsyncMock) as mock_post,
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart())
            body = await resp.json()

    assert resp.status == 503
    assert body["error"]["code"] == "provider_config_error"
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_missing_model_is_rejected_before_runtime_resolution():
    adapter = _make_adapter()
    app = _create_app(adapter)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_runtime:
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart(model=None))
            body = await resp.json()

    assert resp.status == 400
    assert body["error"]["code"] == "missing_model"
    mock_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_missing_file_is_rejected_before_runtime_resolution():
    adapter = _make_adapter()
    app = _create_app(adapter)
    form = FormData()
    form.add_field("model", "whisper-1")
    form.add_field("ignored", b"not-a-file-field", filename="ignored.txt", content_type="text/plain")
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_runtime:
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=form)
            body = await resp.json()

    assert resp.status == 400
    assert body["error"]["code"] == "missing_file"
    mock_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_empty_file_is_rejected():
    adapter = _make_adapter()
    app = _create_app(adapter)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_runtime:
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart(audio=b""))
            body = await resp.json()

    assert resp.status == 400
    assert body["error"]["code"] == "invalid_audio"
    mock_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_oversized_file_is_rejected(monkeypatch):
    adapter = _make_adapter()
    app = _create_app(adapter)
    monkeypatch.setattr(api_server_mod, "AUDIO_TRANSCRIPTION_MAX_BYTES", 4)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_runtime:
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart(audio=b"12345"))
            body = await resp.json()

    assert resp.status == 413
    assert body["error"]["code"] == "audio_too_large"
    mock_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_unsupported_audio_format_is_rejected():
    adapter = _make_adapter()
    app = _create_app(adapter)
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider") as mock_runtime:
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/audio/transcriptions",
                data=_multipart(filename="voice.txt", content_type="text/plain"),
            )
            body = await resp.json()

    assert resp.status == 400
    assert body["error"]["code"] == "unsupported_content_type"
    mock_runtime.assert_not_called()


@pytest.mark.asyncio
async def test_upstream_failure_returns_structured_error():
    adapter = _make_adapter()

    async def fake_post(**_kwargs):
        return None, web.json_response(
            api_server_mod._openai_error(
                "Upstream transcription failed: model unavailable.",
                err_type="server_error",
                code="upstream_transcription_failed",
            ),
            status=502,
        )

    app = _create_app(adapter)
    with (
        patch("hermes_cli.runtime_provider.resolve_requested_provider", return_value="custom"),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={"provider": "custom", "base_url": "http://proxy/v1", "api_key": "proxy-key"},
        ),
        patch.object(adapter, "_post_audio_transcription", side_effect=fake_post),
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/audio/transcriptions", data=_multipart())
            body = await resp.json()

    assert resp.status == 502
    assert body["error"]["code"] == "upstream_transcription_failed"


@pytest.mark.asyncio
async def test_transcription_endpoint_requires_api_key_when_configured():
    adapter = _make_adapter(api_key="sk-test")
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/v1/audio/transcriptions", data=_multipart())
        assert resp.status == 401
