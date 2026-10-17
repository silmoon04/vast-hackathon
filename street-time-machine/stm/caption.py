"""caption: clothing colours and carried items of the one person in each crop, from a vision LLM, for Find & Follow.

describe_crops(paths) sends each crop (base64 JPEG data URL) to an OpenAI-compatible chat completions endpoint
(STM_FIND_LLM, default the local relay http://127.0.0.1:8782/v1/chat/completions, which adds auth on its side: no
key is ever in this code) and returns, per crop, in order:
  {"desc": "red hoodie, black jeans, grey backpack", "look": {"top": "red", "bottom": "black", "items": ["backpack"]},
   "ok": bool, "person": bool|None, "error": str|None}
The model is asked for clothing colours and carried items only; never identity, age, gender, ethnicity, face or
body. Anything like that in the reply is dropped word by word before it is returned. ok=False (desc "") on any
failure (timeout, HTTP error, unreadable crop, reply with no usable JSON) so the caller falls back to its own
colour estimate. A crop with no person gives ok=True, person=False, desc "".
Colour and garment words are normalised to the ones the Find & Follow search understands (grey, navy, t-shirt,
trousers, backpack, ...).
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8782/v1/chat/completions"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
MAX_TOKENS = 120

SYSTEM = (
    "You describe ONLY the clothing colours and carried items of the single person in a CCTV crop, for a lost-and-found"
    " desk. Never guess or mention identity, name, age, gender, ethnicity, skin, hair, face or body. Answer with one"
    " JSON object and nothing else.")
PROMPT = (
    'If there is a person, answer {"person": true, "top": "<colour> <garment>", "bottom": "<colour> <garment>",'
    ' "items": ["<colour> <item>"]}, e.g. {"person": true, "top": "red hoodie", "bottom": "black jeans",'
    ' "items": ["grey backpack"]}. Use plain colour words (red, blue, green, black, white, grey, yellow, orange, pink,'
    ' purple, brown, beige, navy). items = bags or things carried (backpack, bag, tote, suitcase, umbrella), [] if'
    ' none. Use null for anything you cannot see. If there is no person, answer {"person": false}.')

COLOURS = ("red", "blue", "green", "black", "white", "grey", "yellow", "orange", "pink", "purple", "brown", "beige",
           "navy")
COLOUR_SYN = {"gray": "grey", "silver": "grey", "charcoal": "grey", "maroon": "red", "burgundy": "red",
              "crimson": "red", "scarlet": "red", "khaki": "beige", "tan": "beige", "cream": "beige", "camel": "beige",
              "olive": "green", "teal": "green", "turquoise": "blue", "cyan": "blue", "violet": "purple",
              "lilac": "purple", "lavender": "purple", "magenta": "pink", "gold": "yellow", "mustard": "yellow",
              "golden": "yellow"}
# the search's own synonyms (ui/find.html SYN) plus a few the model tends to use
WORD_SYN = {"tee": "t-shirt", "tshirt": "t-shirt", "rucksack": "backpack", "hood": "hoodie", "hoody": "hoodie",
            "pants": "trousers", "trouser": "trousers", "sweater": "hoodie", "jumper": "hoodie",
            "sweatshirt": "hoodie", "handbag": "bag", "purse": "bag", "satchel": "bag", "beanie": "hat",
            "knapsack": "backpack"}
PHRASE_SYN = {"t shirt": "t-shirt", "tote bag": "tote", "shopping bag": "bag", "shoulder bag": "bag",
              "baseball cap": "cap", "dark blue": "navy", "navy blue": "navy"}
SHADES = {"dark", "light", "pale", "bright", "deep", "dull", "faded", "muted"}
# words that describe the person rather than what they wear or carry: dropped from every reply
BANNED = {"man", "men", "woman", "women", "male", "female", "boy", "boys", "girl", "girls", "guy", "lady", "gentleman",
          "kid", "child", "teen", "teenager", "young", "old", "elderly", "adult", "baby", "toddler", "youth", "person",
          "people", "he", "she", "his", "her", "their", "skin", "skinned", "face", "facial", "hair", "haired", "beard",
          "bearded", "moustache", "mustache", "bald", "blonde", "blond", "brunette", "ginger", "redhead", "eyes",
          "glasses", "sunglasses", "tattoo", "tattooed", "tall", "short", "fat", "thin", "skinny", "slim", "heavy",
          "asian", "african", "caucasian", "hispanic", "latino", "latina", "arab", "white-skinned", "dark-skinned",
          "ethnic", "ethnicity", "race", "wearing", "wears", "dressed", "in", "a", "an", "the", "with", "and", "of",
          "possibly", "maybe", "probably", "likely", "unknown", "unclear", "none", "null", "n/a"}
BAGS = ("backpack", "bag", "tote", "suitcase")

FENCE_RE = re.compile(r"```(?:json)?", re.I)
THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.I)
NO_PERSON_RE = re.compile(r"\bno (?:person|people|one|human)\b|\bnobody\b|\bperson\"?\s*:\s*false", re.I)


def _post(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _data_url(path: str) -> str:
    raw = Path(path).read_bytes()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _body(path: str, model: str) -> dict[str, Any]:
    return {"model": model, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": [{"type": "text", "text": PROMPT},
                                                      {"type": "image_url", "image_url": {"url": _data_url(path)}}]}]}


def _content(resp: dict[str, Any]) -> str:
    """Assistant text of an OpenAI-style response (string or list-of-parts content)."""
    msg = ((resp.get("choices") or [{}])[0] or {}).get("message") or {}
    c = msg.get("content")
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else ""


def parse_json(text: str) -> dict[str, Any] | None:
    """First JSON object in a reply: bare, fenced, or inside prose. None when there is none."""
    text = FENCE_RE.sub(" ", THINK_RE.sub(" ", text or ""))
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _norm(s: Any) -> tuple[str | None, str]:
    """('red', 'red hoodie') from one free-text clothing / item phrase; colour None when it names none."""
    if isinstance(s, dict):
        s = " ".join(str(s.get(k) or "") for k in ("colour", "color", "garment", "item", "type", "name"))
    if not isinstance(s, str):
        return None, ""
    s = re.sub(r"[^a-z\- ]", " ", s.lower().replace("_", " "))
    s = re.sub(r"\s+", " ", s).strip()
    for k, v in PHRASE_SYN.items():
        s = re.sub(rf"\b{k}\b", v, s)
    colour, words = None, []
    for w in s.split():
        w = COLOUR_SYN.get(w, WORD_SYN.get(w, w))
        if w in BANNED or w in SHADES:
            continue
        if w in COLOURS:
            if colour is None:
                colour = w
            if w in words:
                continue
        words.append(w)
    return colour, " ".join(words[:4])


def _from_reply(obj: dict[str, Any]) -> dict[str, Any]:
    person = obj.get("person", obj.get("has_person", True))
    if isinstance(person, str):
        person = person.strip().lower() not in ("false", "no", "none", "0")
    if obj.get("no_person") is True:
        person = False
    if not person:
        return {"desc": "", "look": {"top": None, "bottom": None, "items": []}, "ok": True, "person": False,
                "error": None}
    parts, look = [], {"top": None, "bottom": None, "items": []}
    for key, alts, default in (("top", ("upper", "shirt", "jacket"), "top"), ("bottom", ("lower", "legs"), "bottoms")):
        val = next((obj[k] for k in (key, *alts) if obj.get(k)), None)
        colour, phrase = _norm(val)
        if not phrase:
            continue
        look[key] = colour
        parts.append(phrase if phrase != colour else f"{colour} {default}")
    items = obj.get("items") or obj.get("carried") or []
    if isinstance(items, str):
        items = [x for x in re.split(r",|\band\b", items)]
    for it in items if isinstance(items, list) else []:
        colour, phrase = _norm(it)
        noun = " ".join(w for w in phrase.split() if w not in COLOURS)
        if not noun or noun in ("no", "nothing", "no bag") or noun in look["items"]:
            continue
        look["items"].append(noun)
        parts.append(phrase)
        if len(look["items"]) >= 3:
            break
    desc = ", ".join(parts)
    return {"desc": desc, "look": look, "ok": bool(desc), "person": True, "error": None if desc else "empty reply"}


def describe_one(path: str, *, url: str | None = None, model: str = DEFAULT_MODEL, timeout: float = 8) -> dict[str, Any]:
    """One crop -> {"desc", "look", "ok", "person", "error", "ms"}. Never raises."""
    t0 = time.perf_counter()
    fail = {"desc": "", "look": {"top": None, "bottom": None, "items": []}, "ok": False, "person": None}
    try:
        body = _body(path, model)
    except OSError as e:
        return dict(fail, error=f"read: {type(e).__name__}", ms=0.0)
    try:
        resp = _post(url or os.environ.get("STM_FIND_LLM") or DEFAULT_URL, body, timeout)
        text = _content(resp)
    except Exception as e:   # timeout, refused, HTTP error, bad JSON body: the caller falls back
        return dict(fail, error=f"request: {type(e).__name__}", ms=round((time.perf_counter() - t0) * 1000, 1))
    obj = parse_json(text)
    if obj is None:
        out = (dict(fail, ok=True, person=False, error=None) if NO_PERSON_RE.search(text or "")
               else dict(fail, error="no JSON in reply"))
    else:
        try:
            out = _from_reply(obj)
        except (AttributeError, TypeError, ValueError):
            out = dict(fail, error="bad reply shape")
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def describe_crops(paths: list[str], *, url: str | None = None, model: str = DEFAULT_MODEL, timeout: float = 8,
                   workers: int = 4) -> list[dict[str, Any]]:
    """describe_one for every crop, `workers` requests at a time; results in the order of `paths`."""
    if not paths:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(paths)))) as ex:
        return list(ex.map(lambda p: describe_one(p, url=url, model=model, timeout=timeout), paths))
