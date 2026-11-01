"""Quantity extraction: digits, decimals, fractions, number words, dozen/half/quarter."""
import re

from nlp.unit_extractor import canonical_unit, is_unit

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
}
FRACTION_WORDS = {"half": 0.5, "quarter": 0.25}
SKIPPABLE = {"a", "an", "of", "the"}

_DIGIT = re.compile(r"^\d+(?:\.\d+)?$")
_FRACTION = re.compile(r"^(\d+)/(\d+)$")
_LOOKS_NUMERIC = re.compile(r"^[\d.,/]*\d[\d.,/]*$")


def parse_number(token):
    """Return a float for a numeric token / number word, else None."""
    t = str(token).lower()
    if _DIGIT.match(t):
        return float(t)
    m = _FRACTION.match(t)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    return float(NUMBER_WORDS[t]) if t in NUMBER_WORDS else None


def is_malformed_number(token):
    """True for things like '1.2.3' or '5//2' that look numeric but are not."""
    return bool(_LOOKS_NUMERIC.match(str(token))) and parse_number(token) is None


def _result(qty=None, unit=None, indices=(), text="", flags=None, raw_unit=None,
            unit_inferred=False):
    return {"quantity": qty, "unit": unit, "indices": set(indices), "text": text,
            "flags": flags or set(), "raw_unit": raw_unit, "unit_inferred": unit_inferred}


def _apply_unit(qty, unit):
    """'dozen' is converted to a count of pieces (12 per dozen)."""
    if unit == "dozen":
        return qty * 12, "piece"
    return qty, unit


def find_quantity(tokens):
    """Find the first quantity expression in a list of lower-case tokens.

    Returns a dict: quantity, unit, indices (consumed token positions),
    text (matched phrase), flags (invalid/unsupported markers).
    """
    n = len(tokens)
    for i, tok in enumerate(tokens):
        # half / quarter  (half a kilo, half kg, quarter kg, half dozen)
        if tok in FRACTION_WORDS:
            j = i + 1
            while j < n and tokens[j] in SKIPPABLE and j - i <= 2:
                j += 1
            if j < n and is_unit(tokens[j]):
                q, u = _apply_unit(FRACTION_WORDS[tok], canonical_unit(tokens[j]))
                return _result(q, u, range(i, j + 1), " ".join(tokens[i:j + 1]))
            continue
        # article + unit  (a packet of sugar, a dozen eggs, a kilo of onions)
        if tok in ("a", "an") and i + 1 < n and is_unit(tokens[i + 1]):
            q, u = _apply_unit(1.0, canonical_unit(tokens[i + 1]))
            return _result(q, u, (i, i + 1), " ".join(tokens[i:i + 2]))
        # bare 'dozen'
        if tok in ("dozen", "dozens", "doz"):
            return _result(12.0, "piece", (i,), tok)
        num = parse_number(tok)
        if num is not None:
            flags = set()
            if num <= 0:
                flags.add("non_positive")
            j = i + 1
            if j < n and is_unit(tokens[j]):
                q, u = _apply_unit(num, canonical_unit(tokens[j]))
                return _result(q, u, (i, j), " ".join(tokens[i:j + 1]), flags)
            if j + 1 < n and tokens[j + 1] == "of" and tokens[j].isalpha():
                flags.add("unsupported_unit")
                return _result(num, None, (i, j), " ".join(tokens[i:j + 1]), flags,
                               raw_unit=tokens[j])
            # plain count, e.g. "six apples": unit is inferred as pieces
            return _result(num, "piece", (i,), tok, flags, unit_inferred=True)
        if is_malformed_number(tok):
            idx = (i, i + 1) if i + 1 < n and is_unit(tokens[i + 1]) else (i,)
            return _result(None, None, idx, tok, {"malformed"})
    return _result()
