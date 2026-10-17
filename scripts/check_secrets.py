"""Fail if staged (or given) files contain credentials. Run before every commit (no args, the hook) and with
--all before zipping or pushing a submission.

    .venv/Scripts/python.exe scripts/check_secrets.py            # staged blobs + untracked-but-not-ignored files
    .venv/Scripts/python.exe scripts/check_secrets.py --all      # every tracked + untracked-but-not-ignored file
    .venv/Scripts/python.exe scripts/check_secrets.py path ...   # explicit files/dirs

With no args the STAGED content is scanned (what `git commit` records), not the working copy, so already
committed files are not rescanned; --all reads the working copy of every file git would ship.
A directory arg inside a git work tree is listed by git (.gitignore applies); elsewhere, or when git lists
nothing (an ignored folder), it is walked, skipping VCS, venv, cache and build folders below it.
Prints file:line and the KIND of hit only, never the matched value. A line that holds no secret can carry the
marker `secret-scan: allow` (in a comment) to be skipped. Unreadable files and files over MAX_BYTES count as
failures, except a .pdf over MAX_BYTES (noted and passed: its text is usually compressed) and the image,
video, audio, weight, compressed-archive and Office suffixes in SKIP_SUFFIX, which are never read. An
uncompressed .tar is scanned member by member (hits print as tar:member:line); elsewhere a run of NUL bytes
counts as one line break.
Credential names are any NAME ending in PASSWORD, PASSWD, SECRET(_KEY), ACCESS_KEY, API_KEY or TOKEN (APP_TOKEN,
HF_TOKEN, ...); in env-style files also _PASS, _PW, _PWD and _KEY (a _KEY value counts when it is 16+ letters
and digits). How an UNQUOTED value is read depends on the file:
- env-style files (.env*, *.config/.cfg/.ini/.toml/.yaml, shell/batch scripts, Dockerfile), and in ANY file a
  line that starts with export/readonly/local/declare, %env/%set_env, set/setx, ENV/ARG or a YAML list item
  (`- NAME=value`), or NAME after -e/--env/--from-literal=/echo: a literal unless it expands something (one
  whole $X, starts with $(.., %X% or %(x)s, contains ${...}, or is one whole call such as lookup(x)). A longer
  value starting with $X (`$HOME/pw`) expands only in .sh/.ps1 scripts, Dockerfiles and on those prefixed
  lines other than set/setx and `- NAME=` (batch files expand only %X%);
- source code (.py, .js, .ts, ... see CODE_SUFFIX) and PowerShell `$env:NAME = ...` lines: never a literal
  (a bare word there is a variable, call or number), so only quoted values count;
- anything else (.md, .txt, .ipynb, no suffix): a literal unless it looks like code (attribute access, a call
  or subscript, `name,`) or, after a colon, a type (`password: SecretStr`, `token: string`) or the first word
  of a description (`api_key: NVIDIA_API_KEY from ...`).
A quoted value that is itself a name (`"APP_TOKEN"`, `"X-App-Token"`; not in env-style files) or a special
token (`"[MASK]"`, `"__start__"`) is not a secret. Also flagged in any file: JWTs, bearer tokens, `NAME: type =
"literal"`, os.environ[...]/setdefault/setenv/putenv/SetEnvironmentVariable with a literal, credentials in a
URL (scheme://user:pw@host) or in `-u user:pw`, AWS, HF, W&B, NVIDIA (nvapi-) and OpenAI/Anthropic-style (sk-)
keys, private keys. A 40-hex value after @, rev:, commit, comment= or changeset is a git sha, not a key.
Optional: git config core.hooksPath .githooks  (see .githooks/pre-commit)
"""
from __future__ import annotations

import codecs
import io
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Callable


def _quoted(keys: str, n: int) -> re.Pattern[str]:
    """`"<any>_<key>": "<literal>"` in JSON, dicts, kwargs, YAML, os.environ[...] = ...; `\\"` inside .ipynb JSON.
    The key starts a word: unanchored, a long run of word characters (a hex or base64url blob) took
    quadratic time."""
    return re.compile(r"(?i)(?<![\w-])[\w-]*?(?:" + keys + r")(?:[-_]?key)?\\?['\"]?\]?\s*[:=]\s*"
                      r"(?P<q>\\?['\"])(?P<v>[^\s'\"]{" + str(n) + r",})(?P=q)")


# any NAME ending in a credential word: VSS_PASSWORD, APP_TOKEN, GPU_BEARER_TOKEN, WANDB_API_KEY, S3_SECRET_KEY, ...
_NAMES = r"(?:[A-Z0-9_]*?(?:PASSWORD|PASSWD|SECRET(?:_KEY)?|ACCESS_KEY|API_KEY|TOKEN))"
# value groups: `v` quoted, `u` unquoted (maybe code, see _real), `vl` always a literal, `vk` a literal that must
# mix letters and digits
_VALUE = r"(?:\\?['\"](?P<v>[^\s'\"]{6,})|(?P<u>[^\s'\"#]{6,}))"
# a line prefix after which the value is read as in an env-style file, in any file (`$env:` lines: code)
_PRE = (r"(?P<pre>(?:export|readonly|local|declare(?:\s+-\w+)*)\s+|%(?:set_)?env\s+|@?set\s+\"?|(?:ENV|ARG)\s+"
        r"|\$env:)")
_DECL = r"(?:export\s+)?(?:const|let|var)\s+"  # JS/TS declarations
# `vss_password: str = "..."`, `TOKEN: Final = "..."`. The lookahead keeps `\s*` and the class from sharing
# a run of spaces, and {1,80} bounds the lazy scan: unbounded, `TOKEN:` + 2000 spaces took 31 s.
_ANNOT = r"(?:\s*:\s*(?=[\w.\[|])[\w.\[\], |]{1,80}?(?=\s*=))?"
_LINE = r"(?im)^\s*['\"]?"
RULES = [
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+(?P<v>[A-Za-z0-9._\-]{20,})")),
    # NAME=value at line start (.env, shell, Python, a notebook source line, batch `set`, Dockerfile `ENV`/`ARG`)
    ("assigned secret", re.compile(_LINE + r"(?:" + _PRE + "|" + _DECL + r")?" + _NAMES + _ANNOT
                                   + r"\s*(?P<sep>[=:])\s*" + _VALUE)),
    # `ENV NAME value` (Dockerfile), `%env NAME value` (IPython), `setx NAME value` (batch)
    ("assigned secret", re.compile(_LINE + r"(?P<pre>%?ENV\s+|@?setx\s+)" + _NAMES + r"\s+" + _VALUE)),
    # a later pair on a Dockerfile ENV line; a YAML list item (docker compose `environment:`)
    ("assigned secret", re.compile(r"(?im)^\s*(?P<pre>ENV\s+(?:\S+\s+)+?)" + _NAMES + r"=" + _VALUE)),  # not .*?\s
    ("assigned secret", re.compile(r"(?im)^\s*(?P<pre>-\s+['\"]?)" + _NAMES + r"\s*=\s*" + _VALUE)),
    # docker run -e, kubectl create secret --from-literal=, echo NAME=... >> .env
    ("assigned secret", re.compile(r"(?i)(?<![\w-])(?P<pre>(?:-e|--env)(?:\s+|=)['\"]?|--from-literal=|echo\s+['\"]?)"
                                   + _NAMES + r"=" + _VALUE)),
    ("assigned secret", re.compile(r"(?i)(?:\benviron\[|\b(?:setdefault|setenv|putenv|SetEnvironmentVariable)\()"
                                   r"\s*\\?['\"]" + _NAMES + r"\\?['\"]\s*(?:\]\s*=|,)\s*\\?['\"](?P<v>[^\s'\"\\]{6,})")),
    # machine-made tokens/keys are long; shorter values there are names ("next_token": "<NextToken>")
    ("quoted secret", _quoted(r"password|passwd|secret", 6)),
    ("quoted secret", _quoted(r"token|api[-_]?key|access[-_]?key", 16)),
    ("token in url", re.compile(r"(?i)[?&](?:token|access_token|X-Amz-Signature)=(?P<v>[A-Za-z0-9%._\-]{12,})")),
    ("credentials in url", re.compile(r"(?i)(?<![\w+.-])[a-z][\w+.-]*://(?P<user>[^\s/:@'\"]*):"
                                      r"(?P<vl>[^\s/@'\"]{6,200})@")),
    ("credentials in url", re.compile(r"(?<![\w-])(?:-u|--user)(?:\s+|=)['\"]?(?P<user>[^\s:'\"`]+):"
                                      r"(?P<vl>[^\s/@'\"`]{6,200})")),  # curl -u user:pw
    ("aws-style key", re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY-----")),
    ("nvidia api key", re.compile(r"\bnvapi-(?P<vl>[A-Za-z0-9_-]{30,})")),
    ("openai-style key", re.compile(r"\bsk-(?:proj-|ant-[a-z0-9]+-)?(?=[\w-]{0,100}[A-Z])(?=[\w-]{0,100}\d)"
                                    r"(?P<vl>[A-Za-z0-9_-]{20,})")),
    ("hf/wandb key", re.compile(r"\b(?:hf_[A-Za-z0-9]{30,}|(?P<sha>[0-9a-f]{40})(?=\s*$))", re.M)),
    ("hf/wandb key", re.compile(r"(?i)\bkey\s*[:=]\s*\\?['\"](?P<v>[0-9a-f]{40})\\?['\"]")),  # wandb.login(key=..)
]
# env-style files only: shorter password names, and *_KEY with a long letters-and-digits value (not a path)
ENV_RULES = [
    ("assigned secret", re.compile(_LINE + r"(?:" + _PRE + r")?[A-Z0-9_]*?_(?:PASS|PWD|PW)\s*(?P<sep>[=:])\s*"
                                   + _VALUE)),
    ("assigned secret", re.compile(_LINE + r"(?:" + _PRE + r")?[A-Z0-9_]*?_KEY\s*[=:]\s*\\?['\"]?"
                                   r"(?P<vk>[A-Za-z0-9_+=-]{16,})(?=\\?['\"]?(?:\s|#|$))")),
]
# never read; not .tar (scanned member by member) or .pdf (can hold plain text; noted, not failed, over MAX_BYTES).
# Office files are zip containers like .zip. Not .key: that is also a PEM private key's usual suffix.
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".ico",
               ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".flv",
               ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg",
               ".pt", ".pth", ".ckpt", ".safetensors", ".engine", ".plan", ".onnx", ".tflite", ".gguf", ".h5", ".bin",
               ".npy", ".npz", ".parquet", ".pkl",
               ".zip", ".gz", ".tgz", ".bz2", ".xz", ".zst", ".7z", ".rar", ".jar", ".whl", ".woff", ".woff2",
               ".pptx", ".docx", ".xlsx", ".odp", ".odt", ".ods"}
ALLOW_MARK = "secret-scan: allow"
MAX_BYTES = 5_000_000  # larger files fail the scan ("too large to scan") rather than pass unread; a .pdf is noted
ENV_STYLE_SUFFIX = {".env", ".config", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".sh", ".ps1", ".bat", ".cmd"}
SHELL_SUFFIX = {".sh", ".ps1"}  # with Dockerfiles: `$X` inside a longer value expands there (cmd.exe: only %X%)
# source code: an unquoted value is a variable, call, number or expression, never a literal
CODE_SUFFIX = {".py", ".pyi", ".pyw", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".mts", ".cts", ".tsx", ".vue",
               ".svelte", ".java", ".kt", ".kts", ".scala", ".groovy", ".go", ".rs", ".rb", ".php", ".c", ".h",
               ".cc", ".cpp", ".cxx", ".hpp", ".cs", ".swift", ".dart", ".lua", ".jl", ".r"}
# whole value only: `<pw>`, `${PW...`, `$PW`, `$env:PW`, `$(cmd`, `%PW%`, `{pw}`, `xxxx`, `hf_xxxx`, AWS doc keys,
# `mock-password`, `EMPTY`-style keys for local OpenAI-compatible servers
_PLACEHOLDER = re.compile(
    r"(?i)<[^<>]*>?|\$\{.*|\$(?:env:)?\w+|\$\(.*|`.*|%\w+%|\{.*\}|(?:[a-z]{2,4}[-_]?)?(?:x{4,}|\*{3,}|\.{3,}|0{6,})"
    r"|\S*EXAMPLE(?:KEY)?|(?:your[-_ ]?)?(?:example|changeme|change[-_]me|mock|dummy|fake|placeholder|redacted"
    r"|none|empty|unused|not[-_]needed|password|secret|token)(?:[-_ ]?(?:password|secret|token|key|value|here))*")
# a quoted value that names something (ENV_TOKEN = "APP_TOKEN", HEADER = "X-App-Token") or is a special token
_NAMED = re.compile(r"(?i)[\w.-]*[_.-](?:(?:password|passwd|secret|token)(?:[-_]?key)?|(?:api|access)[-_]?key)")
_SPECIAL_TOKEN = re.compile(r"\W?(?:\[\w+\]|<[^<>\s]+>)|__\w+__")  # [MASK], __start__, "<PRE>" after a SentencePiece mark
# other files (.md, .txt, .ipynb, ...): an unquoted value that reads/computes the secret: attribute access on a
# lowercase name (os.environ[, cfg.x, settings.x), a call or subscript (getenv(, _pick(env, Optional[str]),
# a variable passed on (password=password,)
_CODE = re.compile(r"[a-z_]+\.[A-Za-z_]|[A-Za-z_][\w.]*[(\[](?:[A-Za-z_)\]]|$)|[A-Za-z_]\w*[,)]+$")
_TYPE_WORD = re.compile(r"(?i)(?:string|boolean|integer|number|object|optional|required|unknown|undefined)[.,;?]?")
_DESCRIPTION = re.compile(r"\(?[A-Za-z_]+\)?[.,;]?")  # `client_secret: optional string ...`: words follow
# env-style files: a value that starts with an expansion (%X%, %(x)s) or is one whole call (lookup(team_pw), or
# `lookup(` cut at a quote); `${` anywhere also counts. A '(' or '$' elsewhere is literal. A value that is one whole
# $X, or starts with $( or ${, is a _PLACEHOLDER everywhere; a longer value starting with $X counts only in .sh/.ps1
# scripts and Dockerfiles (_SHELL_EXPANSION): vastkit's env-file reader expands nothing, so `$Abc!9` in a .env (or
# a .bat) is a literal.
_ENV_EXPANSION = re.compile(r"%\w+%|%\(|[A-Za-z_][\w.]*\((?:[A-Za-z_$][^()]*\))?$")
_SHELL_EXPANSION = re.compile(r"\$[({\w]")
_NO_SHELL_PRE = re.compile(r"(?i)['\"]?@?setx?\s|-\s")  # batch set/setx and YAML list items expand no $X
_DOCKERFILE = re.compile(r"dockerfile(?:\..*)?|.*\.dockerfile")
# a 40-hex value right after these is a git sha (pip `@<sha>`, pre-commit `rev:`, git archive's pax `comment=`)
_SHA_CONTEXT = re.compile(r"(?i)(?:@|\b(?:rev|revision|commit|comment|changeset|sha1?)\b\W{0,3})$")

Source = tuple[str, Callable[[], bytes]]  # (name printed in hits, reader)


class TooLarge(Exception):
    """File or staged blob over MAX_BYTES: not scanned, so the scan fails (a .pdf is only noted)."""


def _git(*args: str, cwd: Path | None = None) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True, cwd=cwd).stdout


def _names(out: bytes) -> list[str]:
    return [os.fsdecode(x) for x in out.split(b"\0") if x]


def _read_file(path: Path) -> bytes:
    if path.stat().st_size > MAX_BYTES:
        raise TooLarge
    return path.read_bytes()


def _read_blob(root: Path, path: str) -> bytes:
    data = _git("show", f":0:{path}", cwd=root)
    if len(data) > MAX_BYTES:
        raise TooLarge
    return data


def files_from_git(all_files: bool = False) -> list[Source]:
    """Staged blobs (index content, not the working copy) + untracked-but-not-ignored working files.

    all_files: the working copy of every tracked + untracked-but-not-ignored file instead.
    """
    root = Path(os.fsdecode(_git("rev-parse", "--show-toplevel").strip()))
    others = _git("ls-files", *(["--cached"] if all_files else []), "--others", "--exclude-standard", "-z", cwd=root)
    working = [(p, lambda p=p: _read_file(root / p)) for p in _names(others) if (root / p).is_file()]
    if all_files:
        return working
    staged = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "--ignore-submodules=all", "-z", cwd=root)
    return [(p, lambda p=p: _read_blob(root, p)) for p in _names(staged)] + working


def _junk_dir(name: str) -> bool:
    return (name in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "build",
                     "dist"} or name.startswith((".venv", "venv")) or name.endswith(".egg-info"))


def _dir_files(d: Path) -> list[Path]:
    """git's view of d (tracked + untracked-but-not-ignored); else a walk that skips junk folders below d."""
    try:
        listed = [d / n for n in _names(_git("ls-files", "--cached", "--others", "--exclude-standard", "-z",
                                             "--", ".", cwd=d))]
        if listed:
            return [p for p in listed if p.is_file()]
    except (OSError, subprocess.CalledProcessError):  # no git, or not a work tree
        pass
    found: list[Path] = []
    for top, dirs, files in os.walk(d):
        dirs[:] = sorted(x for x in dirs if not _junk_dir(x))
        found += [Path(top) / f for f in sorted(files)]
    return found


def expand(args: list[str]) -> list[Source]:
    """Explicit file args are always scanned; directory args via ``_dir_files``."""
    paths: list[Path] = []
    for a in args:
        p = Path(a)
        paths += _dir_files(p) if p.is_dir() else [p]
    return [(str(p), lambda p=p: _read_file(p)) for p in paths]


def decode(data: bytes) -> str:
    """BOM-aware (Windows PowerShell 5.1 writes UTF-16); otherwise UTF-8 with replacement, so nothing is skipped."""
    for bom, enc in ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"), (codecs.BOM_UTF8, "utf-8-sig"),
                     (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16")):
        if data.startswith(bom):
            return data.decode(enc, errors="replace")
    return data.decode("utf-8", errors="replace")


def env_style(name: str) -> bool:
    """.env*, *.env, config/shell/batch files and Dockerfiles: values there are literals, not code."""
    base = Path(name).name.lower()
    return (base.startswith(".env") or Path(base).suffix in ENV_STYLE_SUFFIX or bool(_DOCKERFILE.fullmatch(base)))


def shell_style(name: str) -> bool:
    """Shell/batch scripts and Dockerfiles: `$X` starting a longer value is an expansion there."""
    base = Path(name).name.lower()
    return Path(base).suffix in SHELL_SUFFIX or bool(_DOCKERFILE.fullmatch(base))


def code_style(name: str) -> bool:
    """Source code (CODE_SUFFIX): an unquoted value there is code, never a literal."""
    return Path(name.lower()).suffix in CODE_SUFFIX and not env_style(name)


def scan_text(text: str, name: str = "") -> list[tuple[int, str]]:
    """(line, kind) per line with a real-looking secret; first matching kind only. `name` picks the
    unquoted-value rule (``env_style``, ``shell_style``, ``code_style``, else other text)."""
    env = env_style(name)
    shell = env and shell_style(name)
    code = not env and code_style(name)
    rules = RULES + ENV_RULES if env else RULES
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if ALLOW_MARK in line:
            continue
        kind = next((k for k, rx in rules if any(_real(m, env, shell, code) for m in rx.finditer(line))), None)
        if kind:
            hits.append((i, kind))
    return hits


def _tar_members(data: bytes) -> list[tuple[str, bytes]] | None:
    """(name, bytes) of each regular file in an uncompressed tar, read in memory; None if `data` is not one."""
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
            out = []
            for member in t:
                if member.isfile():
                    f = t.extractfile(member)
                    out.append((member.name, f.read() if f else b""))
            return out
    except (tarfile.TarError, EOFError, OSError):
        return None


def scan_source(name: str, data: bytes) -> list[tuple[str, int, str]]:
    """(where, line, kind) hits in one file's bytes. A .tar is scanned member by member (where is
    `<tar>:<member>`, each member read by its own name); otherwise a run of NULs counts as one line break."""
    if Path(name).suffix.lower() == ".tar":
        members = _tar_members(data)
        if members is not None:
            return [(f"{name}:{where}", line, kind) for member, blob in members
                    if Path(member).suffix.lower() not in SKIP_SUFFIX
                    for where, line, kind in scan_source(member, blob)]
    text = re.sub("\0+", "\n", decode(data))
    return [(name, line, kind) for line, kind in scan_text(text, name)]


def _real(m: re.Match[str], env: bool = False, shell: bool = False, code: bool = False) -> bool:
    g = m.groupdict()
    raw = g.get("u") or g.get("v") or g.get("vl") or g.get("vk") or m.group(0)
    value = re.sub(r"(?:\\[nr]|\\)+$", "", raw)  # notebook-escaped line ends
    if len(value) < 6 or _placeholder(value):
        return False
    if g.get("user") is not None and value == g["user"]:  # postgres:postgres-style local defaults
        return False
    if g.get("sha") and _SHA_CONTEXT.search(m.string[max(0, m.start() - 24):m.start()]):
        return False
    if g.get("vk") and not (re.search(r"\d", value) and re.search(r"[A-Za-z]", value)):
        return False
    if g.get("v") and (_SPECIAL_TOKEN.fullmatch(value) or (not env and _NAMED.fullmatch(value))):
        return False
    if not g.get("u"):
        return True
    pre = g.get("pre")
    if pre is not None:  # export/%env/set/ENV/... line: env-style in any file; PowerShell `$env:X = word`: code
        return not pre.startswith("$") and not _env_code(value, shell or not _NO_SHELL_PRE.match(pre))
    if env:
        return not _env_code(value, shell)
    if code:
        return False
    return not _other_code(value, g.get("sep") or "=", m.string[m.end("u"):])


def _env_code(value: str, shell: bool) -> bool:
    """Env-style value that expands something (``_ENV_EXPANSION``; with `shell` also ``_SHELL_EXPANSION``)."""
    return bool(_ENV_EXPANSION.match(value) or (shell and _SHELL_EXPANSION.match(value))) or "${" in value


def _other_code(value: str, sep: str, rest: str) -> bool:
    """Prose/unknown files: `X = os.environ.get(...)`, `x=_pick(env, ...)`, `X = cfg.x`; after a colon a type
    (`password: SecretStr`, `token: string`) or the first word of a description (`api_key: optional string`)."""
    if _CODE.match(value):
        return True
    return sep == ":" and bool(re.fullmatch(r"_?[A-Z][A-Za-z]*", value) or _TYPE_WORD.fullmatch(value) or (
        _DESCRIPTION.fullmatch(value) and re.match(r"\s+[A-Za-z(]", rest)))


def _placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER.fullmatch(value))


def main(argv: list[str]) -> int:
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if any(a.startswith("--") for a in argv) and argv != ["--all"]:
        print("usage: check_secrets.py [--all | PATH ...]  (--all takes no paths; see --help)")
        return 2
    try:
        sources = files_from_git(all_files=True) if argv == ["--all"] else expand(argv) if argv else files_from_git()
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"cannot list files from git ({type(e).__name__}); pass paths explicitly")
        return 2
    bad = unreadable = large = pdfs = 0
    for name, read in sources:
        if Path(name).suffix.lower() in SKIP_SUFFIX:
            continue
        try:
            data = read()
        except TooLarge:
            if Path(name).suffix.lower() == ".pdf":
                print(f"not scanned (PDF over {MAX_BYTES:,} bytes): {name}")
                pdfs += 1
            else:
                print(f"too large to scan: {name}")
                large += 1
            continue
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"{name}: unreadable ({type(e).__name__})")
            unreadable += 1
            continue
        for where, line, kind in scan_source(name, data):
            print(f"{where}:{line}: possible {kind}")
            bad += 1
    problems = ([f"{bad} possible secret(s)"] if bad else []) + ([f"{unreadable} unreadable"] if unreadable else []) \
        + ([f"{large} too large"] if large else [])
    note = f" ({pdfs} large PDF(s) not scanned)" if pdfs else ""
    print(f"scanned {len(sources)} files: {', '.join(problems) or 'OK'}{note}")
    if bad:
        print(f"a line that holds no secret can carry the marker '{ALLOW_MARK}' in a comment, e.g. # {ALLOW_MARK}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
