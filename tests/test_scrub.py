"""Keep scenario terms out of the reusable toolkit and personal home paths out of every committed file.
The toolkit's banned terms are held as digests so this file does not spell them out.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# sha256(term)[:16] by term length, in three forms. Each is looked up at every position of " " + form + " ", so a
# term also matches inside a longer word (hasXyz, noxyz_frames), like the substring patterns these replaced.
#   SUB:  lower case; every run of characters other than a-z, 0-9, '_' and '-' becomes one space (so a term with
#         a space also matches across punctuation: "ab cd" in "Ab: /cd").
#   WORD: whole-word terms, stored as " term ": '-' becomes a space too, like \b with word characters a-z0-9_.
#   RAW:  separator-sensitive terms (drive paths, an address), looked up in the lower-cased line as it is.
# To add a term, merge print(_table("<term in that form>")) into the right table.
SUB: dict[int, frozenset[str]] = {
    5: frozenset({"2ef4d8037dc9501f"}),
    6: frozenset({"b8902a0baa867051", "dd56de4137951d9c", "fc126a81c24191ee"}),
    7: frozenset({"013b6d55eba6e3ea", "1c23ed9310dbb358", "736900ad3b39e850", "7c3439480fffe09d"}),
    8: frozenset({"34e9c9776d070c90", "5d3d318a749ef73c", "5f00368a6ad231c3", "7eb31f336f6e3013", "8fc48fe255209d0b",
                  "d6a6ed178ac032df"}),
    10: frozenset({"e02422c9414539e2"}),
    11: frozenset({"1d44c895bb776b74", "8e658c7b93ce589f", "a99e125069aecc75", "b3ed5f1802d09c13"}),
    12: frozenset({"ef7674bc2d37a00d"}),
    13: frozenset({"5800981fb7c62727", "c47cc9bd23fb97e7"}),
    14: frozenset({"5a95e7f65f8694c0", "99cbce1369f17efd", "dada0d45f4d1f6e1", "e933cc19cc614d62"}),
    15: frozenset({"080e64b795dc50b4"}),
    16: frozenset({"9dd7fb0e2ee0ccf9"}),
}
WORD: dict[int, frozenset[str]] = {
    5: frozenset({"06948ddda1ac396e"}),
}
RAW: dict[int, frozenset[str]] = {
    8: frozenset({"4aaf1d6ed12635e6", "9164f73efcc7daa6", "df8fb33d1f451465"}),
    9: frozenset({"a5e315a4cb60fdf1"}),
    12: frozenset({"6beb526cbbd1e6ab", "6cb2848b4f2d28c3", "c6cf5dea9878a1c6"}),
    13: frozenset({"12aac71c0f9a640b", "ec68d43736ed464c"}),
}
SUB_FOLD, WORD_FOLD = re.compile(r"[^a-z0-9_-]+"), re.compile(r"[^a-z0-9_]+")
FORMS = ((SUB_FOLD, SUB), (WORD_FOLD, WORD), (None, RAW))
BINARY = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".mp4", ".mov", ".wav", ".gz", ".zip", ".pt", ".engine", ".onnx",
          ".parquet", ".woff", ".woff2"}


def _digest(term: str) -> str:
    return hashlib.sha256(term.encode("utf-8")).hexdigest()[:16]


def _table(*terms: str) -> dict[int, frozenset[str]]:
    """Digest table for ``terms`` (already in the form they are looked up in)."""
    out: dict[int, set[str]] = {}
    for t in terms:
        out.setdefault(len(t), set()).add(_digest(t))
    return {n: frozenset(ds) for n, ds in sorted(out.items())}


def _hit(line: str, forms=FORMS) -> str | None:
    """A banned term in ``line`` (in its looked-up form, outer spaces stripped), or None."""
    low = line.lower()
    for fold, digests in forms:
        c = " " + (fold.sub(" ", low) if fold else low) + " "
        for n, ds in digests.items():
            for i in range(len(c) - n + 1):
                if _digest(c[i:i + n]) in ds:
                    return c[i:i + n].strip()
    return None


def _repo_files(root: Path = ROOT) -> list[Path]:
    """Tracked plus untracked-but-not-ignored files: everything the next commit could contain. -z: without it
    git quotes non-ASCII names ("notes-\\303\\251.md"), which are then no file and silently skipped."""
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    return [root / p for p in out.split("\0") if p]


def test_repo_files_lists_non_ascii_names(tmp_path, monkeypatch):
    for var in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE"):
        monkeypatch.delenv(var, raising=False)
    try:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    f = tmp_path / "notes-é.md"
    f.write_text("x\n", encoding="utf-8")
    files = _repo_files(tmp_path)
    assert f in files and all(p.is_file() for p in files)


def test_matcher():
    probe = ((SUB_FOLD, _table("zq-vx_w", "ab cd9")), (WORD_FOLD, _table(" qqx ")), (None, _table("q:/zz", "q:\\zz")))
    hit = lambda line: _hit(line, probe)
    assert hit("see ZQ-vx_wide") == hit("x._zq-vx_w(") == "zq-vx_w"  # case-insensitive, also as a word's start
    assert hit("xzq-vx_w") == hit("const hasZq-vx_wide = 1;") == "zq-vx_w"  # and inside a word (camelCase, glued)
    assert hit('"AB", C:\\cd9') is None and hit("Ab: /cd9") == "ab cd9"  # a run of punctuation = one space
    assert hit("zq vx w") is None  # '_' and '-' are kept, not folded into spaces
    assert hit("a QQX.") == hit("qqx-detection") == hit("cam-qqx-01") == "qqx"  # whole word; '-' is a boundary
    assert all(hit(s) is None for s in ("qqxs", "xqqx", "qqx_ok"))  # but never inside a word
    assert hit("at Q:/zz/x") == "q:/zz" and hit("Q:\\ZZ") == "q:\\zz"  # raw terms keep their separators
    assert hit("q zz") is None and hit("xq-zz") is None


def test_no_scenario_terms_or_personal_paths():
    hits = []
    toolkit_dirs = {"vastkit", "scripts", "deploy", "docs", "tests", "templates", ".githooks"}
    toolkit_files = {"pyproject.toml", ".env.example", ".gitignore"}
    home = Path.home().as_posix().lower().rstrip("/")
    home_pattern = re.compile(re.escape(home) + r"(?:/|$)")
    for path in _repo_files():
        if path.suffix.lower() in BINARY or not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        toolkit = (relative.parts[0] in toolkit_dirs or relative.as_posix() in toolkit_files
                   or (len(relative.parts) == 1 and relative.name.startswith("requirements")
                       and relative.suffix == ".txt"))
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            normalized = re.sub(r"/+", "/", line.lower().replace("\\", "/"))
            if home_pattern.search(normalized):
                hits.append(f"{relative.as_posix()}:{n}: personal home path")
            m = _hit(line) if toolkit else None
            if m:
                hits.append(f"{relative.as_posix()}:{n}: {m!r}")
    assert not hits, "repository scrub failed:\n" + "\n".join(hits)
