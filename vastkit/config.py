"""Team config loader.

Precedence for every key: process env > VAST_ENV_FILE / /config/<team>.config > default.
Aliases are tried in order, env then file for each, so a primary key in the file beats a
fallback alias in the env (file VDB_ENDPOINT beats env S3_ENDPOINT).

Exceptions:
- USERNAME. Windows always sets %USERNAME% to the local account, so on Windows the
  process-env USERNAME is ignored. Use VSS_USERNAME or the config file instead.
- VSS_MOCK=1 replaces the VSS URL and credentials and every GPU URL with the mock's, clears
  GPU_BEARER_TOKEN and the S3/VastDB access and secret keys, and drops VAST_SOCKS.
  MOCK_VSS_URL / MOCK_GPU_URL move the mocks off :8765 / :8766.
- Values are never printed: ``Settings.describe()`` shows only whether each field is set.
"""
from __future__ import annotations

import glob
import os
import shlex
from dataclasses import dataclass, field, fields
from pathlib import Path

MOCK_VSS_URL = "http://127.0.0.1:8765"
MOCK_GPU_URL = "http://127.0.0.1:8766"


def parse_env_file(path: str | os.PathLike) -> dict[str, str]:
    """Parse KEY=VALUE lines (bash style: optional `export`, quotes, # comments).

    A UTF-8 BOM is dropped, and `KEY=   # note` is an empty value, not the comment; `KEY=#x` keeps `#x`
    (no whitespace before the `#`, as in bash).
    """
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key.replace("_", "").isalnum():
            continue
        comment_only = value[:1].isspace() and value.lstrip().startswith("#")
        value = value.strip()
        if comment_only:
            value = ""
        elif value and value[0] in "'\"":
            try:
                value = shlex.split(value, posix=True)[0] if value else ""
            except ValueError:
                value = value.strip("'\"")
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        out[key] = value
    return out


def find_config_file() -> str | None:
    """VAST_ENV_FILE, else the single /config/*.config on the lab VM, else ./.env."""
    explicit = os.environ.get("VAST_ENV_FILE")
    if explicit:
        return explicit if Path(explicit).is_file() else None
    found = sorted(glob.glob("/config/*.config"))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(f"{len(found)} files match /config/*.config; set VAST_ENV_FILE to the one to use")
    if Path(".env").is_file():
        return ".env"
    return None


@dataclass
class Settings:
    # VSS backend
    vss_url: str = ""
    vss_username: str = ""
    vss_password: str = field(default="", repr=False)
    # GPU endpoints (each its own host:port, never derive one from another)
    cosmos_reason_url: str = ""
    cosmos_reason_model: str = ""
    cosmos_embed_url: str = ""
    cosmos_embed_model: str = "nvidia/cosmos-embed1"
    yolo_url: str = ""
    canary_url: str = ""
    canary_model: str = ""
    gpu_bearer_token: str = field(default="", repr=False)
    # Storage
    s3_endpoint: str = ""
    access_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    s3_chunks_bucket: str = ""
    s3_segments_bucket: str = ""
    vdb_endpoint: str = ""
    vastdb_bucket: str = ""
    vdb_schema: str = "vss-schema"
    vdb_collection: str = "vss-collection"
    vdb_prompts_collection: str = "vss-prompts-events"
    # W&B
    wandb_api_key: str = field(default="", repr=False)
    wandb_entity: str = ""
    wandb_project: str = ""
    # Transport
    socks_proxy: str = ""
    mock: bool = False
    source_file: str | None = None

    @property
    def team(self) -> str:
        return self.vss_username

    @property
    def wandb_project_path(self) -> str:
        if self.wandb_entity and self.wandb_project:
            return f"{self.wandb_entity}/{self.wandb_project}"
        return self.wandb_project

    def describe(self) -> dict[str, str]:
        """Safe to print: '<set>' / '<unset>' per field, never a value (URLs and hosts included)."""
        return {f.name: "<set>" if getattr(self, f.name) else "<unset>" for f in fields(self)}


def _pick(env: dict[str, str], file: dict[str, str], *keys: str, default: str = "") -> str:
    """First non-empty value, key by key (primary first), env then file for each key."""
    for k in keys:
        v = env.get(k) or file.get(k)
        if v:
            return v
    return default


def load(env_file: str | None = None) -> Settings:
    path = env_file or find_config_file()
    file_vals = parse_env_file(path) if path else {}
    env = dict(os.environ)
    if os.name == "nt":
        env.pop("USERNAME", None)  # always the Windows account, never the team id

    mock = _pick(env, file_vals, "VSS_MOCK").lower() in {"1", "true", "yes"}
    s = Settings(
        vss_url=_pick(env, file_vals, "VSS_BASE_URL", "VSS_URL", "INGRESS_URL"),
        vss_username=_pick(env, file_vals, "VSS_USERNAME", "USERNAME"),
        vss_password=_pick(env, file_vals, "VSS_PASSWORD", "PASSWORD"),
        cosmos_reason_url=_pick(env, file_vals, "COSMOS3_REASON_URL"),
        cosmos_reason_model=_pick(env, file_vals, "COSMOS3_REASON_MODEL"),
        cosmos_embed_url=_pick(env, file_vals, "COSMOS_EMBED1_URL"),
        cosmos_embed_model=_pick(env, file_vals, "COSMOS_EMBED1_MODEL", default="nvidia/cosmos-embed1"),
        yolo_url=_pick(env, file_vals, "YOLO_URL"),
        canary_url=_pick(env, file_vals, "CANARY_1B_URL"),
        canary_model=_pick(env, file_vals, "CANARY_1B_MODEL"),
        gpu_bearer_token=_pick(env, file_vals, "GPU_BEARER_TOKEN"),
        s3_endpoint=_pick(env, file_vals, "S3_ENDPOINT"),
        access_key=_pick(env, file_vals, "VAST_ACCESS_KEY", "ACCESS_KEY"),
        secret_key=_pick(env, file_vals, "VAST_SECRET_KEY", "SECRET_KEY"),
        s3_chunks_bucket=_pick(env, file_vals, "S3_CHUNKS_BUCKET"),
        s3_segments_bucket=_pick(env, file_vals, "S3_SEGMENTS_BUCKET"),
        vdb_endpoint=_pick(env, file_vals, "VDB_ENDPOINT", "S3_ENDPOINT"),
        vastdb_bucket=_pick(env, file_vals, "VASTDB_BUCKET"),
        vdb_schema=_pick(env, file_vals, "VDB_SCHEMA", default="vss-schema"),
        vdb_collection=_pick(env, file_vals, "VDB_COLLECTION", default="vss-collection"),
        vdb_prompts_collection=_pick(env, file_vals, "VDB_PROMPTS_COLLECTION", default="vss-prompts-events"),
        wandb_api_key=_pick(env, file_vals, "WANDB_API_KEY"),
        wandb_entity=_pick(env, file_vals, "WANDB_ENTITY", "WANDB_TEAM"),
        wandb_project=_pick(env, file_vals, "WANDB_PROJECT"),
        socks_proxy=_pick(env, file_vals, "VAST_SOCKS"),
        mock=mock,
        source_file=path,
    )
    if mock:  # never mix mock and real endpoints, and never tunnel to 127.0.0.1
        s.vss_url = _pick(env, file_vals, "MOCK_VSS_URL", default=MOCK_VSS_URL)
        # Same env vars (and defaults) the VSS mock itself reads.
        s.vss_username = env.get("VSS_MOCK_TEAM") or "team-mock"
        s.vss_password = env.get("VSS_MOCK_PASSWORD") or "mock-password"
        gpu = _pick(env, file_vals, "MOCK_GPU_URL", default=MOCK_GPU_URL)
        for attr in ("cosmos_reason_url", "cosmos_embed_url", "yolo_url", "canary_url"):
            setattr(s, attr, gpu)
        s.gpu_bearer_token = s.access_key = s.secret_key = ""  # no real token goes to a mock or real storage
        s.socks_proxy = ""
    return s


_cached: Settings | None = None


def settings() -> Settings:
    global _cached
    if _cached is None:
        _cached = load()
    return _cached


def reset() -> None:
    global _cached
    _cached = None
