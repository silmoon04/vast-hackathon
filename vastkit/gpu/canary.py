"""Canary-1B ASR client (OpenAI-style POST /v1/audio/transcriptions, multipart).

The model id is uncertain (CANARY_1B_MODEL is intentionally unset and /v1/models 404s on
this NIM), so ``transcribe`` tries the configured id, then 'nvidia/canary-1b', then no id.
Nothing in the default pipeline indexes audio: store results yourself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import http as _http
from .. import video as _video
from ..config import Settings, settings as _settings


def extract_wav(video: str | Path, out: str | Path | None = None, *, rate: int = 16000) -> Path:
    """ffmpeg -> 16 kHz mono PCM WAV (what Riva/Canary expect); without `out`, a temp file the caller deletes."""
    return _video.extract_wav(video, out, rate=rate)


class CanaryASR:
    def __init__(self, base_url: str | None = None, *, model: str | None = None, token: str | None = None,
                 proxy: str | None = None, timeout: float = 180.0, cfg: Settings | None = None):
        cfg = cfg or _settings()
        self.base_url = (base_url or cfg.canary_url).rstrip("/")
        if not self.base_url:
            raise ValueError("CANARY_1B_URL unset (or VSS_MOCK=1 for the GPU mock)")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model or cfg.canary_model or None
        self.client = _http.make_client(self.base_url, proxy=proxy if proxy is not None else cfg.socks_proxy,
                                        token=token if token is not None else cfg.gpu_bearer_token,
                                        timeout=timeout)

    def ready(self) -> bool:
        try:
            return self.client.get("/v1/health/ready").status_code == 200
        except Exception:
            return False

    def transcribe(self, audio: str | Path, *, language: str | None = "en",
                   response_format: str | None = None) -> dict[str, Any]:
        """Returns the server JSON (at least {'text': ...}). Pass a video to auto-extract audio (a temp WAV,
        deleted afterwards)."""
        path = Path(audio)
        tmp: Path | None = None
        if path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            path = tmp = extract_wav(path)
        try:
            last = None
            for model in [m for m in (self.model, "nvidia/canary-1b", None)]:
                data: dict[str, str] = {}
                if model:
                    data["model"] = model
                if language:
                    data["language"] = language
                if response_format:
                    data["response_format"] = response_format
                with path.open("rb") as fh:
                    r = _http.request(self.client, "POST", "/v1/audio/transcriptions", data=data,
                                      files={"file": (path.name, fh, "audio/wav")})
                if r.status_code == 200:
                    self.model = model
                    try:
                        return r.json()
                    except ValueError:
                        return {"text": r.text}
                last = r
                if r.status_code not in (400, 404, 422):
                    break
            assert last is not None
            raise _http.APIError(last, "audio/transcriptions")
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)

    def close(self) -> None:
        self.client.close()
