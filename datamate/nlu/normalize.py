"""Rewrite teencode, abbreviations and Telex typos into plain Vietnamese.

    rewrite("be oi th10/2025 top 3 mon ban nhiu nhut la j z ha")
    -> "be oi tháng 10/2025 top 3 mon ban nhiêu nhất là gì vậy ha"

Best effort, shown to the model next to the original question; never replaces it.
"""

from __future__ import annotations

import re
import unicodedata

# whole-token replacements; ambiguous ones ("hong" = hồng/không, "nhiêu") are left out on purpose
_WORDS = {
    "dthu": "doanh thu",
    "bnhiu": "bao nhiêu",
    "bnhieu": "bao nhiêu",
    "bn": "bao nhiêu",
    "nhiu": "nhiêu",
    "dc": "được",
    "đc": "được",
    "dk": "được",
    "đk": "được",
    "k": "không",
    "ko": "không",
    "kh": "khách hàng",
    "khum": "không",
    "hok": "không",
    "j": "gì",
    "z": "vậy",
    "r": "rồi",
    "sp": "sản phẩm",
    "nv": "nhân viên",
    "ncc": "nhà cung cấp",
    "đh": "đơn hàng",
    "dh": "đơn hàng",
    "tm": "tiền mặt",
    "ck": "chuyển khoản",
    "hnay": "hôm nay",
    "hnai": "hôm nay",
    "hqua": "hôm qua",
    "mik": "mình",
    "mk": "mình",
    "wan": "quán",
    "kua": "của",
    "cko": "cho",
    "jum": "giùm",
    "zum": "giùm",
    "nhut": "nhất",
    "nhứt": "nhất",
    "trc": "trước",
    "ng": "người",
    "sl": "số lượng",
    "tg": "thời gian",
    "vs": "với",
    "lun": "luôn",
    "ntn": "như thế nào",
    "kái": "cái",
    "kác": "các",
    "kòn": "còn",
    "tíh": "tính",
    "nhug": "những",
    "nhìu": "nhiều",
    "ni": "này",
    "trog": "trong",
    "đượt": "được",
    "báng": "bán",
}
_PERIOD = (
    (re.compile(r"\b(?:th|t)\s*\.?\s*(\d{1,2})\s*([/-])\s*(\d{2,4})\b", re.I), r"tháng \1\2\3"),
    (re.compile(r"\bq\s*([1-4])\s*([/-])\s*(\d{2,4})\b", re.I), r"quý \1\2\3"),
    (re.compile(r"\bth\s+(này|ni|trước|rồi)\b", re.I), r"tháng \1"),
)
_TELEX_PAIRS = (
    ("dd", "đ"),
    ("uow", "ươ"),
    ("aa", "â"),
    ("aw", "ă"),
    ("ee", "ê"),
    ("oo", "ô"),
    ("ow", "ơ"),
    ("uw", "ư"),
)
_TONES = {"s": "́", "f": "̀", "r": "̉", "x": "̃", "j": "̣"}
_VOWELS = "aăâeêioôơuưy"
_MODIFIED = "ăâêôơư"


def _telex(tok: str) -> str:
    """'ddieenj' -> 'điện', 'nguwowif' -> 'người'. Only tokens with a Telex digraph are touched."""
    low = tok.lower()
    if not re.search(r"dd|aa|aw|ee|oo|ow|uw", low) or not low.isalpha():
        return tok
    tone = ""
    if low[-1] in _TONES and len(low) > 2:
        tone, low = _TONES[low[-1]], low[:-1]
    for a, b in _TELEX_PAIRS:
        low = low.replace(a, b)
    if not tone:
        return low
    vowel_idx = [i for i, ch in enumerate(low) if ch in _VOWELS]
    if not vowel_idx:
        return tok
    modified = [i for i in vowel_idx if low[i] in _MODIFIED]
    if modified:
        pos = modified[-1]
    elif vowel_idx[-1] < len(low) - 1 or len(vowel_idx) == 1:
        pos = vowel_idx[-1]  # closed syllable: tone on the last vowel ("toán")
    else:
        pos = vowel_idx[-2]  # open syllable with two vowels: tone on the first ("của")
    return unicodedata.normalize("NFC", low[: pos + 1] + tone + low[pos + 1 :])


def rewrite(question: str) -> str:
    s = question
    for rx, rep in _PERIOD:
        s = rx.sub(rep, s)
    out = []
    for tok in re.split(r"(\W+)", s):
        low = tok.lower()
        out.append(_WORDS.get(low, _telex(tok)) if tok and tok[0].isalpha() else tok)
    return re.sub(r"\s+", " ", "".join(out)).strip()
