"""Stage 1 of the cascade: answer the questions whose wording maps to a query without any LLM call.

A question qualifies only when every word of it is accounted for: period words, names found in
the shop's data (a product, a product group, a channel…), the cue words of one family, and filler
("cho hỏi", "giùm", "nhỉ"). One unexplained word — a condition, another measure, a name that is
not in the data — and the question is left to the model. Families:

    plain      doanh thu, số đơn, đơn huỷ, tỉ lệ huỷ, giá trị đơn trung bình, ngày cao nhất, % thay đổi
    filtered   doanh thu / số lượng bán của một mặt hàng, một nhóm hàng, một kênh, một nhóm khách…
    share      tỉ trọng của một nhóm / kênh / hình thức thanh toán trong doanh thu cả shop
    ranking    top k mặt hàng bán chạy (số lượng quy về đơn vị cơ bản), top k khách theo doanh thu,
               khách mua nhiều đơn nhất

`python -m eval.oracle --train <train>` runs exactly this code against the train answers; it is how
the sales model, the period rules and every family here are validated. Any wording added must keep
that report at 100 %. The definitions are those of the rulebook rows listed in agent/rules.py: when
a split's rulebook words one of them differently, the family is switched off for the run.

Nothing here depends on a business id or a question id.
"""

from __future__ import annotations

import re

from datamate.agent.query import Query
from datamate.core.periods import Period, prev_period, same_days_prev_month, same_period_last_year, this_month
from datamate.core.text import fold, key
from datamate.nlu.normalize import rewrite
from datamate.nlu.periods import extract

_REVENUE = r"doanh thu|dthu|bao nhieu tien|may tien|nhieu tien|thu ve|thu duoc"
_FILLER = set(
    """
cua hang shop quan tiem doanh thu dthu tong cong ban duoc dc ra bao nhieu nhiu bnhiu nhieu may tien la trong cho
toi minh biet giup xem voi hoi nhe nha a the vay z ha khong ko k oi tinh so don bi huy ti ty le phan tram gia tri
trung binh quan mot moi tren mang ve ngay nao cao nhat dat xet theo thay doi tang giam hay lien truoc thang nam quy
hom nay qua tuan roi dem co da dang tat ca cac lai duoc luong hang ngay hnay hqua bua dat gi sao duoc nhe dum giup
jum cko mk mik e anh chi em ban ah u uh chua hien thi lam on bn nhiu nhieu phan tram kia vua bang tron
""".split()
)
# padding seen around every kind of question; none of these changes what is asked
_FILLER |= set("gium check coi nhi ta do total ee nay ne di revenue bill order o trang thai xin vui long".split())
# for the families with names: politeness, question words and period residue only — the words that
# carry meaning ("huỷ", "trung bình", "doanh thu"…) must come from the family's own cue list
_POLITE = set(
    """
cua hang shop quan tiem cho toi minh biet giup xem voi hoi nhe nha a the vay z ha oi e ee anh chi em ah u uh lam on
gium dum jum check coi nhi ta do di ne nay xin vui long hay la trong duoc dc ra tong cong total tat ca cac nhung cai
bao nhieu nhiu bnhiu may bn thang nam quy tuan hom qua roi truoc ngay vao luc thi
""".split()
)
_PERIOD_WORDS = re.compile(
    r"\b(thang|th|t|quy|q)\s*\d{1,2}(\s*(/|-|nam)\s*\d{2,4})?|\b\d{1,2}/\d{1,2}/\d{4}|\b\d{1,3}\s*ngay\b|\d+"
)
_TONE_KEYS = "sfrxj"
_DOUBLES = (("dd", "d"), ("aa", "a"), ("ee", "e"), ("oo", "o"), ("aw", "a"), ("ow", "o"), ("uw", "u"))


def _detelex(tok: str, vocab: set[str]) -> str:
    """Telex typing left unconverted, undone in accent-free space: 'hangf' -> 'hang', 'toongr' -> 'tong',
    'ddown' -> 'don'. A Vietnamese syllable never ends in s/f/r/x/j, so such a token is a tone key that
    was not applied. Only tokens that become a known word are changed."""
    if tok in vocab or not tok.isalpha() or len(tok) < 3:
        return tok
    cands = [tok]
    if tok[-1] in _TONE_KEYS:
        cands.append(tok[:-1])
    for c in list(cands):
        d = c
        for a, b in _DOUBLES:
            d = d.replace(a, b)
        d = d.replace("w", "")
        cands.append(d)
    for c in cands[1:]:
        if c in vocab:
            return c
    return tok


_VOCAB: set[str] = set()


def _repair(text: str) -> str:
    """Telex slips undone inside a text that keeps its punctuation ("quys 3/2025" -> "quy 3/2025"), for the
    period parser."""
    return re.sub(r"[a-z]+", lambda m: _detelex(m.group(0), _VOCAB), fold(text))


def normalize(question: str) -> str:
    """Accent-free question with teencode rewritten and Telex slips repaired."""
    s = key(rewrite(question))
    s = " ".join(_detelex(t, _VOCAB) for t in s.split())
    return re.sub(r"\bnha thuoc\b", "shop", s)  # the shop itself, not the medicines it sells


def is_plain(s: str) -> bool:
    """s: normalized question."""
    rest = _PERIOD_WORDS.sub(" ", s)
    return all(tok in _FILLER for tok in re.findall(r"[a-z]+", rest))


def intent(s: str, raw: str) -> str | None:
    """s: normalized question; raw: original text (for '%')."""
    percent = "%" in raw or re.search(r"ti le|ty le|phan tram", s)
    if re.search(r"\bhuy\b", s):
        if percent:
            return "cancel_rate"
        if re.search(r"\b(bao nhieu|may|so|dem)\b", s):
            return "cancel_count"
        return None
    if re.search(r"trung binh|binh quan", s) and re.search(r"\b(don|bill|order)\b", s):
        return "avg_order"
    if re.search(r"\bngay nao\b", s) and re.search(r"nhat", s):
        return "best_day"
    if re.search(r"truoc|so voi", s) and percent and re.search(_REVENUE, s):
        return "pct_change"
    if re.search(_REVENUE, s):
        return "revenue"
    if re.search(r"\b(don|bill|order)\b", s) and re.search(r"bao nhieu|may|\bso\b", s):
        return "order_count"
    return None


def base_period(s: str, periods: list[Period]) -> Period:
    """The period a percentage change is measured against."""
    p = periods[0]
    if re.search(r"cung ky nam (truoc|ngoai)", s):
        return same_period_last_year(p)
    if re.search(r"(lien|ngay) truoc|truoc do|thang truoc|ky truoc", s) or len(periods) < 2:
        return same_days_prev_month(p.end) if p == this_month(p.end) else prev_period(p)
    return periods[1]


_QUERY = {
    "revenue": ("value", "revenue"),
    "order_count": ("value", "orders"),
    "cancel_count": ("value", "cancelled_orders"),
    "cancel_rate": ("value", "cancel_rate"),
    "avg_order": ("value", "avg_order_value"),
    "best_day": ("best_day", "revenue"),
    "pct_change": ("change", "revenue"),
}

# ------------------------------------------------------------------ names in the question

LINK_DIMS = ("product", "product_group", "channel", "payment", "customer_group", "staff")
_CUE_BEFORE = {  # a short value ("lẻ", "sỉ", "COD") counts as a name only right after one of these words
    "product_group": ("nhom", "hang"),
    "customer_group": ("nhom",),
    "channel": ("kenh", "qua", "tai"),
    "payment": ("bang", "toan"),
    "staff": ("vien", "nv"),
    "product": (),
}


def _loose(v: str) -> str:
    return key(v).replace(" ", "")


class Names:
    """The shop's values of each linkable dimension, indexed for exact (spacing-insensitive) lookup."""

    def __init__(self, dims: dict[str, list[str]]):
        self.index: dict[str, list[tuple[str, str]]] = {}
        for dim in LINK_DIMS:
            for value in dims.get(dim, ()):
                k = _loose(value)
                if len(k) >= 2 and not k.isdigit():
                    self.index.setdefault(k, []).append((dim, value))

    def link(self, tokens: list[str]) -> list[tuple[int, int, str, str]] | None:
        """Names found in a token list as (first token, end token, dimension, value), longest first at
        each position; None when a name is ambiguous or a dimension is named twice."""
        spans: list[tuple[int, int, str, str]] = []
        seen: set[str] = set()
        i = 0
        while i < len(tokens):
            hit = None
            for j in range(min(len(tokens), i + 14), i, -1):  # longest name first
                span = "".join(tokens[i:j])
                cands = self.index.get(span)
                if not cands:
                    continue
                prev = tokens[i - 1] if i else ""
                ok = [(d, v) for d, v in cands if len(span) >= 5 or prev in _CUE_BEFORE[d]]
                # the word in front settles a name that exists in two dimensions ("nhóm trà" / món "Trà")
                ok = [(d, v) for d, v in ok if prev in _CUE_BEFORE[d]] or ok
                if len({d for d, _ in ok}) > 1:
                    return None
                if ok:
                    hit = (j, *ok[0])
                    break
            if hit is None:
                i += 1
                continue
            j, dim, value = hit
            if dim in seen:
                return None  # two names of one dimension: a comparison or a list, not a simple filter
            seen.add(dim)
            spans.append((i, j, dim, value))
            i = j
        return spans

    def mask(self, question: str) -> tuple[str, dict[str, str]] | None:
        """(question with every name replaced by the token "zqz", {dimension: value}). Done on the text
        as typed, before any rewriting: "Shopee" or "Coopmart" must not be read as Telex slips."""
        toks = [(m.start(), m.end(), key(m.group(0)).replace(" ", "")) for m in re.finditer(r"\w+", question)]
        spans = self.link([k for _, _, k in toks])
        if spans is None:
            return None
        out, pos = [], 0
        for i, j, _dim, _value in spans:
            out += [question[pos : toks[i][0]], " zqz "]
            pos = toks[j - 1][1]
        out.append(question[pos:])
        return "".join(out), {dim: value for _, _, dim, value in spans}


# ------------------------------------------------------------------ families with names

_UNITS = set(
    """
kg ly dia cai hop vien chai lon cay bao goi thung phan tuyp lo mieng cuon binh bo ket noi con qua bich tui vi
m met lit cuc tam to hu loc
""".split()
)
_W_MONEY = set(
    "tien ban thu ve doanh dthu mat hang san pham sp mon thuoc cua tu nhom don khach kenh qua tra bang".split()
)
_W_SHARE = set(
    """
ti trong chiem dong gop phan cua nhom hang doanh thu dthu tien tong ca don ban qua tra bang khach kenh so phan tram
""".split()
)
_W_QTY = set("ban mat hang san pham sp mon so luong".split()) | _UNITS
_W_TOP = set("top liet ke mat hang san pham sp mon thuoc ban chay hot nhat nhieu nao gi loai".split())
_W_CUST = set("top liet ke khach hang ai nguoi mua chi tieu nhieu tien nhat nao don".split())
_TOP_CUE = re.compile(r"ban chay|chay nhat|hot nhat|ban nhieu nhat|ban (duoc|ra) nhieu nhat")
_CUST_MONEY = re.compile(r"(chi tieu|mua( hang)?) nhieu( tien)? nhat|nhieu tien nhat|tieu nhieu nhat")
_CUST_ORDERS = re.compile(r"mua nhieu don( hang)? nhat")
_K = re.compile(r"\btop (\d+)\b|\bliet ke (\d+)\b|\b(\d+) (?:mat hang|san pham|sp|mon|hang|thuoc|khach)\b")

_VOCAB |= _FILLER | _POLITE | _W_MONEY | _W_SHARE | _W_QTY | _W_TOP | _W_CUST
_VOCAB |= set("thang quy nam ngay tuan hom nay qua truoc roi so voi lien cung ky dau het den tu gan nhat".split())


# period expressions as they look after key() ("7/2025" -> "7 2025"); what is left of the digits are counts
_PERIOD_EXPR = re.compile(
    r"\b(?:thang|th|t|quy|q) \d{1,2}(?: (?:nam )?\d{2,4})?\b|\b\d{1,2} \d{1,2} \d{4}\b|\b\d{1,3} ngay\b"
    r"|\bnam \d{4}\b|\b\d{1,2} (?:20)?\d{2}\b"
)


def _only(words: list[str], *allowed: set[str]) -> bool:
    ok = set().union(_POLITE, *allowed)
    return all(w in ok or w == "<e>" for w in words)


def _with_names(raw: str, period: Period, names: Names) -> tuple[str, Query] | None:
    masked = names.mask(raw)  # before anything is rewritten or stripped: names contain numbers and doubled letters
    if masked is None:
        return None
    s = normalize(masked[0])
    ents = masked[1]
    tokens = ["<e>" if t == "zqz" else t for t in s.split()]
    text = _PERIOD_EXPR.sub(" ", " ".join(tokens))
    k, as_list = 1, bool(re.search(r"\bliet ke\b|\bnhung\b|\btop\b", text))
    m = _K.search(text)
    if m:
        k, as_list = int(next(g for g in m.groups() if g)), True
        text = text[: m.start()] + m.group(0).replace(str(k), " ", 1) + text[m.end() :]
    if re.search(r"\d", text):
        return None  # a number nobody explained: a threshold, a second period, a quantity…
    words = text.split()
    core = " ".join(words)
    percent = "%" in raw or bool(re.search(r"phan tram", s))

    if not ents:  # rankings name nothing
        if _TOP_CUE.search(core) and _only(words, _W_TOP):
            return "top_products", Query("top", "base_qty", period, by="product", k=k, as_list=as_list)
        if re.search(r"\bkhach\b|\bai\b", core) and _only(words, _W_CUST):
            if _CUST_ORDERS.search(core) and k == 1:
                return "customer_most_orders", Query("top", "orders", period, by="customer", k=1, as_list=as_list)
            if _CUST_MONEY.search(core) and not re.search(r"\bdon\b", core):
                return "top_customers", Query("top", "revenue", period, by="customer", k=k, as_list=as_list)
        return None

    if len(ents) != 1 or m:
        return None
    (dim, _value), filters = next(iter(ents.items())), dict(ents)
    if percent:
        if (
            dim != "product"
            and re.search(r"ti trong|chiem|dong gop|phan cua|bang bao nhieu", core)
            and re.search(_REVENUE, core)
            and _only(words, _W_SHARE)
        ):
            return f"{dim}_share", Query("share", "revenue", period, filters=filters)
        return None
    if dim == "product":
        unit = re.search(r"\b(?:may|bao nhieu|so) (\w+)\b", core)
        if unit and unit.group(1) in _UNITS and not re.search(r"\btien\b", core) and _only(words, _W_QTY):
            return "product_qty", Query("value", "base_qty", period, filters=filters)
    if re.search(_REVENUE + r"|tien ban", core) and _only(words, _W_MONEY):
        if dim in ("product", "product_group") or re.search(r"\bdon\b", core):
            return f"{dim}_revenue", Query("value", "revenue", period, filters=filters)
    return None


def resolve(question: str, as_of, cal=None, names: Names | None = None) -> tuple[str, Query] | None:
    """(family, query) for a question this stage can answer, None when it needs a model.
    names: the shop's Names; without it only plain questions are recognized."""
    periods = extract(_repair(rewrite(question)), as_of, cal)
    if not periods:
        return None
    s = normalize(question)
    kind = intent(s, question)
    if kind is not None and is_plain(s):
        op, metric = _QUERY[kind]
        return kind, Query(op, metric, periods[0], base_period(s, periods) if kind == "pct_change" else None)
    if names is not None and len(periods) == 1:
        return _with_names(question, periods[0], names)
    return None
