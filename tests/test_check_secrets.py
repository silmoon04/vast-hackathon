"""scripts/check_secrets.py and .githooks/pre-commit. Every fake secret is built at runtime, so this file scans clean."""
import base64
import importlib.util
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_secrets", ROOT / "scripts" / "check_secrets.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

PW = "Sup3r" + "S3cretPw"
LEAK = 'PASSWORD = "' + PW + '"\n'  # a literal in every file class (in source code only quoted values count)
TOK = "q8ZrLm" + "2Vx9Tk"
HUNTER = "Hunter" + "2026x"
HAS_GIT = shutil.which("git") is not None


def _jwt() -> str:
    part = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")
    return ".".join(part(b) for b in (json.dumps({"alg": "HS256"}).encode(), b'{"sub":"team-test"}', os.urandom(24)))


def _hits(line: str, name: str = "") -> list[str]:
    return [kind for _, kind in cs.scan_text(line, name)]


def _ids(cases: list) -> list[str]:
    """Stable test ids that never contain the (partly random) fake secret."""
    return [f"case{i}" for i in range(len(cases))]


FLAGGED = [
    "VSS_PASSWORD=" + "Fake." + "Pass99",
    'VSS_PASSWORD="' + "Fake." + 'Pass99"',
    "PASSWORD=" + "Hack(2026)" + "xyz",
    "PASSWORD=" + "P@ss$" + "WORD99",
    "PASSWORD=" + "example" + "sandbox9",
    '{"username": "team-7", "pass' + 'word": "' + PW + '"}',
    'cfg = {"secret' + '_key": "' + "wJalrXUtnFEMIK7MDENG" + 'bPxRfiCYzzzzzzzzzz"}',
    'wandb.login(key="' + secrets.token_hex(20) + '")',
    'os.environ["VSS_PASS' + 'WORD"] = "' + PW + '"',
    '{"token": "${T}", "pass' + 'word": "' + PW + '"}',  # a placeholder earlier on the line hides nothing
    "TOKEN = '" + _jwt() + "'",
    # the same lines as .ipynb JSON source strings
    '    "os.environ[\\"WANDB_API_KEY\\"] = \\"' + secrets.token_hex(20) + '\\"\\n",',
    '    "{\\"pass' + 'word\\": \\"' + PW + '\\"}\\n",',
    '    "VSS_PASSWORD=' + PW + '\\n",',
    "%env VSS_PASSWORD=" + PW,
]
# ignored in source code and in other text files (some are literals in env-style files, see ENV_FLAGGED)
IGNORED = [
    "    client_secret: optional string sent with the request",  # docstring prose
    "    api_key: _AzureApiKeyT | None,",
    'VSS_PASSWORD = os.environ.get("VSS_PASSWORD", "")',
    '        vss_password=_pick(env, file_vals, "VSS_PASSWORD", "PASSWORD"),',
    "PASSWORD = cfg.vss_password",
    "SECRET_KEY = settings.secret_key",
    'VSS_PASSWORD="${VSS_PASSWORD:-${PASSWORD:-}}"',
    "PASSWORD=$(cat /run/team-pw)",
    "PASSWORD=<team password>",
    "VSS_PASSWORD=",
    'cfg = Settings(vss_password="mock-password")',
    '"secret_key": "{secret_key}"',
    '"token_type": "bearer", "tokenizer": "bert-base", "secretName": "team-creds"',
    "HF_TOKEN=hf_" + "x" * 34,
    "VSS_PASSWORD: Optional[str] = None",
    'export VSS_PASSWORD="$(kubectl get secret team-creds -o name)"',
    'client = OpenAI(base_url=url, api_key="not-needed")',
    '    "VSS_PASSWORD = os.environ[\\"VSS_PASSWORD\\"]\\n",',
    '    "PASSWORD=<team password>\\n",',
    # a quoted value that is a name or a special token
    'ENV_TOKEN = "APP_TOKEN"',
    'API_KEY = "WANDB_API_KEY"',
    'HEADER_TOKEN = "X-App-Token"',
    'EXPIRED_TOKEN = "expired_token"',
    'ADLS_SAS_TOKEN = "adls.sas-token"',
    'VIDEO_TOKEN = "[VIDEO]"',
    "mask_token='[MASK]'",
    'bos_token="__start__"',
    'prefix_token="' + chr(0x2581) + '<PRE>",',  # a SentencePiece special token
    "Markdown: run `curl -u user:password` first",  # backtick closes inline code; `password` is a placeholder
    'NGC_KEY = "nvapi-' + "x" * 40 + '"',
    "pkg @ git+https://example.org/x.git@" + "3f78" * 10,  # a 40-hex git sha, not a W&B key
    "commit " + "3f78" * 10,
    "DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/db",  # user == password: a local default
    "VSS_URL=https://team7:${VSS_PASSWORD}@vss.example.org",
    "curl -u $VSS_USERNAME:$VSS_PASSWORD http://127.0.0.1:1/api",
]
# source code: an unquoted value is code (a variable, call, number or expression), never a literal
CODE_IGNORED = [
    "    token = authorization[7:] if authorization else None",  # a slice, not a literal
    "        next_api_key = effective_api_key",  # a variable passed on
    "    api_key = API_KEY_SENTINEL",
    "    webhook_secret=webhook_secret or self.webhook_secret,",
    "    get_token: Callable[[], str]",
    # one line of each false-positive class seen in site-packages
    "next_token = (logits[:, -1, :] / temperature).argmax(-1)",
    "token = (await resp.json())['access_token']",
    "api_key = (os.environ.get('NVIDIA_API_KEY') or os.environ.get('NGC_API_KEY'))",
    "id_to_token = {v: k for k, v in vocab.items()}",
    "token = -logits.max()",
    "token = AddedToken(**token)",
    "VIDEO_TOKEN = 151656",
    "    text_pad_token=10_048,",
    "token: OAuth2Token | None = None",
    "secret: V1Secret | None = None",
    "token = result",
    "access_token = response",
    "export const API_TOKEN = readToken();",
]
CODE_FLAGGED = [
    'PASSWORD = "' + PW + '"',
    'APP_TOKEN = "' + TOK + '"',
    'vss_password: str = "' + HUNTER + '"',  # an annotation between the name and '='
    'APP_TOKEN: Final = "' + TOK + '"',
    'os.environ.setdefault("VSS_PASSWORD", "' + HUNTER + '")',
    'monkeypatch.setenv("APP_TOKEN", "' + TOK + '")',
    'os.environ["APP_TOKEN"] = "' + TOK + '"',
    'NGC_KEY = "nvapi-' + secrets.token_urlsafe(32) + '"',
    'url = "https://team7:' + HUNTER + '@vss.example.org"',
    "cmd = 'docker run -e VSS_PASSWORD=" + HUNTER + " img'",
    'const APP_TOKEN = "' + TOK + '";',
]


@pytest.mark.parametrize("line", FLAGGED, ids=_ids(FLAGGED))
def test_flags_literal_secrets(line):
    assert _hits(line), line


@pytest.mark.parametrize("name", ["", "app.py", "notes.md"])
@pytest.mark.parametrize("line", IGNORED, ids=_ids(IGNORED))
def test_ignores_code_and_placeholders(line, name):
    assert not _hits(line, name), (name, line)


@pytest.mark.parametrize("name", ["app.py", "web/app.ts"])
@pytest.mark.parametrize("line", CODE_IGNORED, ids=_ids(CODE_IGNORED))
def test_code_files_ignore_unquoted_values(line, name):
    assert not _hits(line, name), (name, line)


@pytest.mark.parametrize("name", ["app.py", "web/app.ts"])
@pytest.mark.parametrize("line", CODE_FLAGGED, ids=_ids(CODE_FLAGGED))
def test_code_files_flag_quoted_literals(line, name):
    assert _hits(line, name), (name, line)


# env-style files: an unquoted value is a literal unless it expands something ($X, %X%, ${..}, a call)
ENV_FLAGGED = [
    (".env", "VSS_PASSWORD=" + "london" + ".Bridge123"),
    ("team.env", "VSS_PASSWORD=" + "summer" + ".breeze77"),
    (".env.example", "PASSWORD=" + "Hunter" + "2026,"),
    ("team.config", "PASSWORD = cfg" + ".vss_password"),  # code in a .py file, a literal here
    ("deploy/run.sh", "export VSS_PASSWORD=" + "london" + ".Bridge123"),
    ("values.yaml", "password: " + HUNTER),
    # a '(' that is not one whole call, or a '$' / '%' that starts no expansion, is part of the literal
    (".env", "VSS_PASSWORD=" + "9xK(" + "2mQ!7z"),
    (".env", "VSS_PASSWORD=" + "Hunter" + "(2026)"),
    (".env", "SECRET_KEY=" + "abcDEF" + "(123)ghi456"),
    ("team.config", "PASSWORD=" + "Qw3rty" + "(uiop"),
    (".env", "VSS_PASSWORD=" + "%Tr0ub" + "4dor99"),
    (".env", "VSS_PASSWORD=" + "$!x9" + "Tr0ub4dor"),
    # `$X` starting a longer value expands only in shell scripts and Dockerfiles: a literal in .env/config/YAML
    (".env", "VSS_PASSWORD=" + "$Abc" + "9xK!z"),
    ("team.config", "PASSWORD=" + "$Hunter" + "2026!"),
    ("values.yaml", "password: " + "$Tr0ub" + "4dor!"),
    # Dockerfile and batch syntax
    ("deploy/app/Dockerfile", "ENV VSS_PASSWORD=" + HUNTER),
    ("Dockerfile", "ENV VSS_PASSWORD " + HUNTER),
    ("Dockerfile.gpu", "ARG VSS_PASSWORD=" + HUNTER),
    ("Dockerfile", "ENV VSS_USERNAME=team7 VSS_PASSWORD=" + HUNTER),  # a later pair on the line
    ("go.bat", "set VSS_PASSWORD=" + HUNTER),
    ("go.bat", "@set VSS_PASSWORD=" + HUNTER),
    ("go.bat", "setx VSS_PASSWORD " + HUNTER),
    ("go.cmd", 'set "VSS_PASSWORD=' + HUNTER + '"'),
    ("go.bat", "set VSS_PASSWORD=" + "$Abc" + "9xK!z"),  # cmd.exe expands %X% only
    # any NAME ending in a credential word, not only the toolkit's own keys
    (".env", "APP_TOKEN=" + TOK),
    ("team.env", "MY_SERVICE_API_KEY=" + "k3yV4l" + "ue9z"),
    ("deploy/run.sh", "export DB_PASSWD=" + HUNTER),
    # env-style only: *_PASS / *_PW / *_PWD, and *_KEY with a long letters-and-digits value; key formats
    (".env", "VSS_PASS=" + HUNTER),
    (".env", "DB_PWD=" + HUNTER),
    (".env", "SERVICE_KEY=" + secrets.token_hex(16)),
    (".env", "NGC_KEY=nvapi-" + secrets.token_urlsafe(32)),
    (".env", "OPENAI_KEY=sk-proj-" + "AbCdEf" + "Gh1jKl" + secrets.token_hex(12)),
    # docker compose list items, docker run -e, kubectl --from-literal, echo into a file, shell builtins
    ("docker-compose.yml", "      - VSS_PASSWORD=" + HUNTER),
    ("docker-compose.yml", '      - "APP_TOKEN=' + TOK + '"'),
    ("run.sh", "docker run -e VSS_PASSWORD=" + HUNTER + " img"),
    ("run.sh", "kubectl create secret generic creds --from-literal=VSS_PASSWORD=" + HUNTER),
    ("run.sh", "echo VSS_PASSWORD=" + HUNTER + " >> .env"),
    ("run.sh", "readonly APP_TOKEN=" + TOK),
    ("run.sh", "declare -x VSS_PASSWORD=" + HUNTER),
    # PowerShell: only quoted values (a bare word there is a command)
    ("run.ps1", '$env:APP_TOKEN = "' + TOK + '"'),
    ("run.ps1", '[Environment]::SetEnvironmentVariable("VSS_PASSWORD", "' + HUNTER + '", "User")'),
    # credentials in a URL or a -u option
    (".env", "VSS_URL=https://team7:" + HUNTER + "@vss.example.org"),
    ("run.sh", "curl -u team7:" + HUNTER + " http://127.0.0.1:1/api"),
]
ENV_IGNORED = [
    (".env", "PASSWORD=$(cat /run/team-pw)"),
    (".env", 'VSS_PASSWORD="${VSS_PASSWORD:-}"'),
    ("run.sh", "VSS_PASSWORD=pre${PW}"),
    (".env", "VSS_PASSWORD=pre${PW}"),
    (".env", "VSS_PASSWORD=$TEAM_PW"),  # one whole $X, any env-style file
    ("run.sh", "PASSWORD=$HOME/team-pw"),
    ("deploy/run.sh", "export VSS_PASSWORD=$PW_BASE.x"),
    ("Dockerfile", "ENV VSS_PASSWORD=$PW_BASE/x"),
    ("run.cmd", "PASSWORD=%TEAM_PW%"),
    ("app.ini", "password = %(base_pw)s"),
    ("app.cfg", "PASSWORD=lookup(team_pw)"),
    ("app.cfg", 'PASSWORD=lookup("team")'),
    (".env", "PASSWORD=<team password>"),
    (".env", "VSS_PASSWORD="),
    (".env", "VSS_PASSWORD=changeme"),
    ("Dockerfile", "ENV VSS_PASSWORD=${VSS_PASSWORD}"),
    ("Dockerfile", "ENV VSS_USERNAME=team7 VSS_PASSWORD=${VSS_PASSWORD}"),
    ("Dockerfile", "ARG VSS_PASSWORD"),
    ("go.bat", "set VSS_PASSWORD=%TEAM_PW%"),
    ("run.ps1", "$env:VSS_PASSWORD = Read-Host -AsSecureString"),
    ("run.sh", "docker run -e VSS_PASSWORD=$VSS_PASSWORD img"),
    ("docker-compose.yml", "      - VSS_PASSWORD=${VSS_PASSWORD}"),
    # *_KEY values that are paths, column names or short; placeholders; special tokens; git shas
    (".env", "S3_KEY=videos/clip.mp4"),
    (".env", "SORT_KEY=created_at_timestamp"),
    (".env", "PUBLIC_KEY=ssh-rsa AAAAB3NzaC1yc2E"),
    (".env", "NGC_KEY=nvapi-" + "x" * 40),
    ("model.yaml", 'mask_token: "[MASK]"'),
    (".pre-commit-config.yaml", "    rev: " + "3f78" * 10),
]


@pytest.mark.parametrize("name,line", ENV_FLAGGED, ids=_ids(ENV_FLAGGED))
def test_env_style_unquoted_values_are_literals(name, line):
    assert cs.scan_text(line, name), (name, line)


@pytest.mark.parametrize("name,line", ENV_IGNORED, ids=_ids(ENV_IGNORED))
def test_env_style_expansions_and_placeholders_are_ignored(name, line):
    assert not cs.scan_text(line, name), (name, line)


# other text (.md, .txt, .ipynb): a literal unless it looks like code, a type or the start of a description;
# export/%env/set/ENV/ARG lines read their value as in an env-style file
OTHER_FLAGGED = [
    ("nb.ipynb", '    "%env VSS_PASSWORD=' + "team_pw" + '_2026\\n",'),
    ("nb.ipynb", '    "%env VSS_PASSWORD=' + "london" + '.Bridge123\\n",'),
    ("nb.ipynb", '    "%set_env APP_TOKEN=' + TOK + '\\n",'),
    ("nb.ipynb", '    "%env VSS_PASSWORD ' + HUNTER + '\\n",'),
    ("notes.md", "VSS_PASSWORD=" + "team_pw" + "_2026"),
    ("notes.md", "export VSS_PASSWORD=" + "team_pw" + "_2026"),
    ("notes.md", "export VSS_PASSWORD=" + "london" + ".Bridge123"),
    ("notes.md", "PASSWORD=" + "TEAM7_SECRET" + "_2026"),
    ("notes.md", "password: " + "sun" + "shine"),
    ("notes.md", "password: " + "hunter" + "bridge"),
    ("notes.md", "VSS_PASSWORD=" + "Pa[55]" + "word"),
    ("notes.md", "- VSS_PASSWORD=" + HUNTER),
    ("notes.md", "VSS_URL=https://team7:" + HUNTER + "@vss.example.org"),
    ("notes.txt", "kubectl create secret generic creds --from-literal=VSS_PASSWORD=" + HUNTER),
    ("notes.txt", "NVIDIA_KEY=nvapi-" + secrets.token_urlsafe(32)),
]
OTHER_IGNORED = [
    ("notes.md", "APP_TOKEN: required for writes"),
    ("notes.md", "TOKEN: (optional) a bearer token"),
    ("notes.md", "api_key: NVIDIA_API_KEY from the build page"),
    ("notes.md", "token: string;"),
    ("notes.md", "- APP_TOKEN=<random>"),
    ("notes.md", "export VSS_PASSWORD=$PW_BASE.x"),
    ("nb.ipynb", '    "%env VSS_PASSWORD=$TEAM_PW\\n",'),
    ("notes.md", "VSS_PASS=" + HUNTER),  # short password names count in env-style files only
]


@pytest.mark.parametrize("name,line", OTHER_FLAGGED, ids=_ids(OTHER_FLAGGED))
def test_other_text_literals_are_flagged(name, line):
    assert cs.scan_text(line, name), (name, line)


@pytest.mark.parametrize("name,line", OTHER_IGNORED, ids=_ids(OTHER_IGNORED))
def test_other_text_code_and_prose_are_ignored(name, line):
    assert not cs.scan_text(line, name), (name, line)


def test_code_files_keep_code_heuristic():
    line = "PASSWORD = cfg" + ".vss_password"
    assert not cs.scan_text(line, "app.py") and not cs.scan_text(line, "web/app.js") and not cs.scan_text(line)
    assert cs.scan_text(line, ".env")


@pytest.mark.parametrize("name,expected", [
    (".env", True), (".env.local", True), ("sub/team.env", True), ("/config/team-7.config", True),
    ("x.toml", True), ("x.yml", True), ("tunnel.ps1", True), ("go.bat", True), ("Dockerfile", True),
    ("Dockerfile.gpu", True), ("app.py", False), ("app.js", False), ("notes.txt", False), ("environment.py", False),
])
def test_env_style_names(name, expected):
    assert cs.env_style(name) is expected


@pytest.mark.parametrize("name,expected", [
    ("run.sh", True), ("tunnel.ps1", True), ("go.bat", False), ("go.cmd", False), ("Dockerfile", True),
    ("app.dockerfile", True), (".env", False), ("team.config", False), ("x.yaml", False), ("app.py", False),
])
def test_shell_style_names(name, expected):
    assert cs.shell_style(name) is expected


@pytest.mark.parametrize("name,expected", [
    ("app.py", True), ("web/App.TSX", True), ("web/app.ts", True), ("main.go", True), ("notes.md", False),
    ("nb.ipynb", False), ("notes.txt", False), ("", False), (".env", False), ("run.sh", False),
])
def test_code_style_names(name, expected):
    assert cs.code_style(name) is expected


def test_tar_and_pdf_are_scanned(tmp_path, capsys):
    """An uncompressed tar is scanned member by member, and a PDF can hold plain text: both are read."""
    data = f"VSS_PASSWORD={PW}\n".encode("utf-8")
    tar = tmp_path / "bundle.tar"
    with tarfile.open(tar, "w") as t:
        info = tarfile.TarInfo("conf/team.env")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF-1.4\n" + data)
    assert cs.main([str(tar)]) == 1
    out = capsys.readouterr().out
    assert "bundle.tar:conf/team.env:1: possible assigned secret" in out and PW not in out
    assert cs.main([str(pdf)]) == 1
    out = capsys.readouterr().out
    assert "notes.pdf:2: possible assigned secret" in out and PW not in out


def test_git_archive_tar_header_is_not_a_key(tmp_path, capsys):
    """`git archive` writes a pax header `comment=<commit sha>`: a header, not a member, and a sha, not a key."""
    tar = tmp_path / "submission.tar"
    with tarfile.open(tar, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": secrets.token_hex(20)}) as t:
        data = b"print('hello')\n"
        info = tarfile.TarInfo("repo/main.py")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    assert cs.main([str(tar)]) == 0, capsys.readouterr().out


def test_nul_runs_count_as_one_line_break(tmp_path, capsys):
    f = tmp_path / "blob.dat"
    f.write_bytes(b"head" + b"\0" * 512 + f"VSS_PASSWORD={PW}\n".encode("utf-8"))
    assert cs.main([str(f)]) == 1
    assert "blob.dat:2: possible assigned secret" in capsys.readouterr().out


@pytest.mark.parametrize("alphabet", ["0123456789abcdef", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"])
def test_long_blob_line_scans_in_linear_time(alphabet):
    """One 200k-character run of word characters (a hex or base64url blob in .json/.ipynb): was quadratic."""
    line = "".join(secrets.choice(alphabet) for _ in range(200_000))
    t0 = time.perf_counter()
    for name in ("x.json", "x.py", ".env"):
        cs.scan_text(line, name)
    assert time.perf_counter() - t0 < 10


def test_long_space_run_scans_in_linear_time():
    """Spaces after `NAME:` (the annotation rule) or after ENV (the later-pair rule): were cubic and quadratic."""
    lines = ["TOKEN:" + " " * 4000, "password: " + " " * 4000 + "x", "ENV" + " " * 4000 + "X"]
    t0 = time.perf_counter()
    for line in lines:
        for name in ("x.json", ".env", "x.py", "x.md", "run.sh", "Dockerfile", "go.bat"):
            cs.scan_text(line, name)
    assert time.perf_counter() - t0 < 10


def test_annotation_and_later_env_pair_still_flagged():
    value = "Hunter" + "2026xQ"
    assert cs.scan_text(f'TOKEN: dict[str, str] | None = "{value}"\n', "app.py")
    assert cs.scan_text(f'ENV A="x y" VSS_PASSWORD={value}\n', "Dockerfile")


def test_env_file_hit_through_main(tmp_path, capsys):
    value = "london" + ".Bridge123"
    f = tmp_path / ".env"
    f.write_text(f"VSS_URL=http://127.0.0.1:1\nVSS_PASSWORD={value}\n", encoding="utf-8")
    assert cs.main([str(f)]) == 1
    out = capsys.readouterr().out
    assert ".env:2: possible assigned secret" in out and value not in out
    assert cs.ALLOW_MARK in out  # the escape for a false positive is printed with the hits


def test_oversized_file_fails_but_binary_is_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cs, "MAX_BYTES", 16)
    big = tmp_path / "notes.txt"
    big.write_text("x = 1\n" * 10, encoding="utf-8")
    video = tmp_path / "clip.mp4"
    video.write_bytes(os.urandom(64))
    small = tmp_path / "ok.txt"
    small.write_text("x = 1\n", encoding="utf-8")
    assert cs.main([str(video), str(small)]) == 0
    assert "too large" not in capsys.readouterr().out
    assert cs.main([str(big), str(small)]) == 1
    out = capsys.readouterr().out
    assert f"too large to scan: {big}" in out and "1 too large" in out


def test_large_pdf_is_noted_and_office_files_are_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cs, "MAX_BYTES", 16)
    pdf = tmp_path / "pitch.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + os.urandom(64))
    deck = tmp_path / "pitch.pptx"
    deck.write_bytes(b"PK\x03\x04" + os.urandom(64))
    assert cs.main([str(pdf), str(deck)]) == 0
    out = capsys.readouterr().out
    assert f"not scanned (PDF over 16 bytes): {pdf}" in out and "1 large PDF(s) not scanned" in out
    assert "pptx" not in out and "too large" not in out


def test_help_and_unknown_options(capsys):
    assert cs.main(["--help"]) == 0 and "check_secrets.py --all" in capsys.readouterr().out
    assert cs.main(["-h"]) == 0
    capsys.readouterr()
    assert cs.main(["--bogus"]) == 2 and "usage:" in capsys.readouterr().out
    assert cs.main(["--all", "x.py"]) == 2


def test_allow_mark_and_one_hit_per_line():
    assert not _hits(f"PASSWORD={PW}  # {cs.ALLOW_MARK}")
    assert _hits(f'export PASSWORD="{PW}"') == ["assigned secret"]


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32", "utf-8-sig", "cp1252"])
def test_non_utf8_files_are_scanned(tmp_path, capsys, encoding):
    f = tmp_path / "notes.txt"
    f.write_bytes(f"cost £ 3\nVSS_PASSWORD={PW}\n".encode(encoding))  # utf-16/32 add a BOM, as PowerShell 5.1 does
    assert cs.main([str(f)]) == 1
    out = capsys.readouterr().out
    assert "notes.txt:2: possible assigned secret" in out and PW not in out


def test_unreadable_file_fails(tmp_path, capsys):
    assert cs.main([str(tmp_path / "missing.env")]) == 1
    assert "missing.env: unreadable" in capsys.readouterr().out


def test_own_files_scan_clean(capsys):
    assert cs.main([str(ROOT / "scripts" / "check_secrets.py"), __file__]) == 0, capsys.readouterr().out


def _no_git_env(monkeypatch) -> None:
    for var in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        monkeypatch.delenv(var, raising=False)


def _repo(tmp_path: Path, monkeypatch) -> Path:
    _no_git_env(monkeypatch)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_dir_walk_skips_venv_cache_and_build_folders(tmp_path, monkeypatch, capsys):
    _no_git_env(monkeypatch)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))  # not inside any work tree
    for rel in (".venv/x.py", ".venv-embed/Lib/y.py", ".pytest_cache/v/f", "pkg.egg-info/PKG-INFO", "build/lib/z.py",
                "__pycache__/a.txt"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(LEAK, encoding="utf-8")
    assert cs.main([str(tmp_path)]) == 0, capsys.readouterr().out
    (tmp_path / "top.py").write_text(LEAK, encoding="utf-8")
    assert cs.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "top.py:1: possible assigned secret" in out and "scanned 1 files" in out and PW not in out
    assert cs.main([str(tmp_path / ".venv")]) == 1  # an explicit junk folder is still scanned


@pytest.mark.skipif(not HAS_GIT, reason="git not on PATH")
def test_dir_arg_in_repo_follows_gitignore(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path, monkeypatch)
    (repo / ".gitignore").write_text("*.env\nignored/\n", encoding="utf-8")
    for rel in ("sub/leak.py", "sub/team.env", "ignored/x.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(LEAK, encoding="utf-8")
    assert cs.main([str(repo / "sub")]) == 1
    out = capsys.readouterr().out
    assert "leak.py:1:" in out and "team.env" not in out and "scanned 1 files" in out
    assert cs.main([str(repo / "ignored")]) == 1  # git lists nothing in an ignored folder: it is walked
    assert "x.py:1:" in capsys.readouterr().out


@pytest.mark.skipif(not HAS_GIT, reason="git not on PATH")
def test_all_scans_working_copy_of_tracked_and_untracked(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path, monkeypatch)
    (repo / "staged.py").write_text(LEAK, encoding="utf-8")
    (repo / "dirty.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "staged.py", "dirty.py")
    (repo / "staged.py").write_text('PASSWORD = os.environ["P"]\n', encoding="utf-8")  # clean working copy
    (repo / "dirty.py").write_text(LEAK, encoding="utf-8")  # tracked, edited, not staged
    (repo / "new.txt").write_text(f"PASSWORD={PW}\n", encoding="utf-8")
    (repo / ".gitignore").write_text("*.env\n", encoding="utf-8")
    (repo / "team.env").write_text(f"PASSWORD={PW}\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    assert cs.main(["--all"]) == 1
    out = capsys.readouterr().out
    assert "dirty.py:1:" in out and "new.txt:1:" in out and "staged.py" not in out and "team.env" not in out


@pytest.mark.skipif(not HAS_GIT or not (ROOT / ".git").exists(), reason="needs git and a checkout")
def test_repo_scans_clean(monkeypatch, capsys):
    _no_git_env(monkeypatch)
    monkeypatch.chdir(ROOT)
    assert cs.main(["--all"]) == 0, capsys.readouterr().out


@pytest.mark.skipif(not HAS_GIT, reason="git not on PATH")
def test_no_args_scans_staged_blob_not_working_copy(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path, monkeypatch)
    (repo / "leak.py").write_text(f'TOKEN = "{_jwt()}"\n', encoding="utf-8")
    (repo / "gone.py").write_text(LEAK, encoding="utf-8")
    (repo / "dirty.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "leak.py", "gone.py", "dirty.py")
    (repo / "leak.py").write_text('TOKEN = os.environ["T"]\n', encoding="utf-8")  # clean working copy
    (repo / "gone.py").unlink()  # staged, deleted from disk
    (repo / "dirty.py").write_text(LEAK, encoding="utf-8")  # unstaged edit: not committed
    (repo / "new.txt").write_text(f"PASSWORD={PW}\n", encoding="utf-8")  # untracked, not ignored
    (repo / ".gitignore").write_text("*.env\n", encoding="utf-8")
    (repo / "team.env").write_text(f"PASSWORD={PW}\n", encoding="utf-8")  # ignored
    (repo / "sub").mkdir()
    monkeypatch.chdir(repo / "sub")  # paths stay repo-root relative from a subdirectory
    assert cs.main([]) == 1
    out = capsys.readouterr().out
    assert "leak.py:1: possible jwt" in out and "gone.py:1: possible assigned secret" in out
    assert "new.txt:1:" in out and "dirty.py" not in out and "team.env" not in out and PW not in out


@pytest.mark.skipif(not HAS_GIT, reason="git not on PATH")
def test_oversized_staged_blob_fails(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path, monkeypatch)
    (repo / "big.txt").write_text("x = 1\n" * 10, encoding="utf-8")
    (repo / "pic.png").write_bytes(os.urandom(64))
    _git(repo, "add", "big.txt", "pic.png")
    (repo / "big.txt").write_text("x\n", encoding="utf-8")  # the staged blob is what counts
    monkeypatch.setattr(cs, "MAX_BYTES", 16)
    monkeypatch.chdir(repo)
    assert cs.main([]) == 1
    out = capsys.readouterr().out
    assert "too large to scan: big.txt" in out and "pic.png" not in out


@pytest.mark.skipif(not HAS_GIT or shutil.which("sh") is None, reason="needs git and sh")
def test_pre_commit_hook_runs_scanner(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    for rel in ("scripts/check_secrets.py", ".githooks/pre-commit"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, repo / rel)
    # the interpreter running pytest is a working `python` (and `python3` on Linux); a Windows python3 stub is skipped
    env = {**os.environ, "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")}
    hook = lambda: subprocess.run(["sh", ".githooks/pre-commit"], cwd=repo, env=env, capture_output=True, text=True,
                                  encoding="utf-8")
    assert hook().returncode == 0
    (repo / "leak.py").write_text(LEAK, encoding="utf-8")
    _git(repo, "add", "leak.py")
    r = hook()
    assert r.returncode == 1 and "leak.py:1: possible assigned secret" in r.stdout and PW not in r.stdout + r.stderr
