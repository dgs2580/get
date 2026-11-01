"""Rule-based entity (item) extraction on top of NLTK tokens.

Pipeline per request:
  sentences (NLTK) -> strip request phrases ("please buy ...") -> split into
  item segments -> tokenise each segment (NLTK) -> quantity/unit extraction ->
  item phrase = remaining content words -> normalisation.

Quantities are attached ONLY to the segment they were written in, so
"2 kg rice, 1 litre milk" can never mix up the pairs.
"""
import re

from nlp.normalizer import display_name, normalize_item
from nlp.preprocessing import clean_text, is_word_token, split_sentences, tokenize
from nlp.quantity_extractor import NUMBER_WORDS, find_quantity
from nlp.unit_extractor import format_number

# ---- request phrases that are NOT shopping items ---------------------------
_OCCASION = (r"for (?:the |this |next |my |our |a )?(?:weekend|week|month|day|party|dinner|lunch|"
             r"breakfast|tonight|tomorrow|today|trip|barbecue|bbq|picnic|birthday|home|house|kitchen)")
_LEAD = re.compile(
    r"^(?:please|kindly|pls|can you|could you|would you|will you|i need to|i need|"
    r"i want to|i want|i would like to|i would like|i'd like to|i'd like|i wish to|"
    r"we need|we want|we have to|i have to|i must|i should|let's|lets|"
    r"also|and also|and|plus|then|add|buy|get|purchase|pick up|grab|order|"
    r"put|include|bring|me|us|" + _OCCASION + r")\b[\s,:;]*", re.I)
_TRAIL = re.compile(
    r"[\s,]*\b(?:(?:to|on|in|into)\s+(?:my|the|our)\s+(?:shopping\s+|grocery\s+)?"
    r"(?:list|cart|basket)|please|thanks|thank you|pls)\b[\s.!?]*$", re.I)
_HEADER = re.compile(r"^(?P<h>.+?)\s+(?:like|such as|including|for example|e\.g\.)\s+(?P<t>.+)$", re.I)
_SPLIT = re.compile(r"\s*(?:,|;|\band\b|&|\bplus\b|\bas well as\b|\balso\b)\s*", re.I)
_PROTECTED = ["mac and cheese", "fish and chips"]
_AND_HALF = re.compile(r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+and\s+a\s+half\b", re.I)

# Words dropped from the item phrase once quantity/unit have been removed
FILLER = {"some", "any", "a", "an", "the", "of", "my", "our", "to", "list", "please",
          "around", "about", "approx", "approximately", "roughly", "nearly", "almost",
          "more", "extra", "another", "few", "couple", "lot", "lots", "like", "also",
          "me", "us", "for", "or", "with", "add", "buy", "get", "need", "want"}
NON_ITEM_WORDS = {"hello", "hi", "hey", "thanks", "thank", "ok", "okay", "yes", "no", "bye",
                  "sure", "well", "hmm", "test", "nothing"}
MAX_ITEM_WORDS = 4


def _strip_leads(text):
    prev = None
    while prev != text:
        prev = text
        text = _LEAD.sub("", text.strip()).strip()
        text = _TRAIL.sub("", text).strip()
    return text.strip(" .,:;!?")


def segment_sentence(sentence):
    """Return (segments, ignored_header_phrases) for one sentence."""
    ignored = []
    s = sentence.strip()
    s = _AND_HALF.sub(lambda m: _half(m.group(1)), s)
    for p in _PROTECTED:
        s = re.sub(re.escape(p), p.replace(" and ", " _AND_ "), s, flags=re.I)
    s = _strip_leads(s)
    # "household items: soap, shampoo"  /  "vegetables like tomatoes, onions"
    if ":" in s:
        head, _, tail = s.partition(":")
        if tail.strip():
            ignored.append(head.strip())
            s = tail.strip()
    else:
        m = _HEADER.match(s)
        if m and not re.search(r"\d", m.group("h")):
            ignored.append(m.group("h").strip())
            s = m.group("t")
    parts = [p.replace("_AND_", "and").strip() for p in _SPLIT.split(s)]
    return [p for p in parts if p], ignored


def _half(n):
    n = n.lower()
    val = float(NUMBER_WORDS[n]) if n in NUMBER_WORDS else float(n)
    return format_number(val + 0.5)


def parse_segment(segment):
    """Parse one segment into an entry dict (or None when nothing is left)."""
    original = segment.strip()
    text = _strip_leads(clean_text(original))
    if not text:
        return None
    flags, notes = set(), []
    negative_text = None
    m = re.search(r"(?<![\w.])-\s*(\d+(?:\.\d+)?)", text)
    if m:  # negative quantity such as "-2 kg rice"
        negative_text = "-" + m.group(1)
        text = text[:m.start()] + m.group(1) + text[m.end():]

    tokens = [t for t in tokenize(text) if is_word_token(t) or t.isdigit()]
    qty = find_quantity(tokens)
    rest = [t for i, t in enumerate(tokens) if i not in qty["indices"]]
    words = [t for t in rest if t not in FILLER and re.search(r"[a-z]", t)]
    phrase = " ".join(words)

    entry = {
        "original_phrase": original, "phrase": phrase,
        "quantity": qty["quantity"], "unit": qty["unit"],
        "unit_inferred": qty["unit_inferred"], "quantity_text": qty["text"],
        "quantity_status": "specified" if qty["quantity"] is not None else "missing",
        "status": "Extracted", "notes": notes,
    }
    has_qty = bool(qty["indices"])

    if negative_text:
        entry.update(quantity=None, unit=None, quantity_status="invalid",
                     status="Invalid quantity", unit_inferred=False)
        notes.append(f"Negative quantity ({negative_text}) is not allowed and was ignored.")
    elif "non_positive" in qty["flags"]:
        entry.update(quantity=None, unit=None, quantity_status="invalid",
                     status="Invalid quantity", unit_inferred=False)
        notes.append("Quantity must be greater than zero; the value was ignored.")
    elif "malformed" in qty["flags"]:
        entry.update(quantity=None, unit=None, quantity_status="invalid",
                     status="Invalid quantity")
        notes.append(f"'{qty['text']}' is not a valid number.")
    elif "unsupported_unit" in qty["flags"]:
        entry.update(unit=None, quantity_status="invalid", status="Unsupported unit")
        entry["quantity"] = None
        notes.append(f"Unit '{qty['raw_unit']}' is not supported; quantity was not assigned.")
    return entry if (phrase or has_qty or negative_text) else None


def build_item(entry):
    """Normalise the phrase and compute display fields."""
    from nlp.category_classifier import classify_item
    from nlp.unit_extractor import display_unit

    normalized = normalize_item(entry["phrase"])
    cat = classify_item(normalized)
    q, u = entry["quantity"], entry["unit"]
    if q is not None:
        quantity_display = f"{format_number(q)} {display_unit(u, q)}".strip()
    else:
        quantity_display = "Not specified"
    confidence = 0.5 + (0.3 * cat["confidence"] if cat["category"] != "Other" else 0.0)
    confidence += 0.1 if q is not None else 0.0
    confidence += 0.1 if (q is not None and not entry["unit_inferred"]) else 0.0
    if entry["status"] != "Extracted":
        confidence -= 0.2
    if len(entry["phrase"].split()) > MAX_ITEM_WORDS:
        confidence -= 0.2
        entry["notes"].append("Long phrase - it may contain more than one item.")
    return {
        "item": normalized,
        "display_name": display_name(normalized),
        "quantity": q,
        "unit": u,
        "unit_display": display_unit(u, q) if u else "Not specified",
        "quantity_display": quantity_display,
        "quantity_status": entry["quantity_status"],
        "unit_inferred": entry["unit_inferred"],
        "category": cat["category"],
        "category_method": cat["method"],
        "confidence": round(max(0.05, min(confidence, 0.99)), 2),
        "status": entry["status"],
        "original_phrase": entry["original_phrase"],
        "quantity_text": entry["quantity_text"],
        "notes": entry["notes"],
    }


def extract_items(text):
    """Extract structured items from natural-language text.

    Returns dict(items, warnings, segments, ignored_phrases).
    """
    cleaned = clean_text(text)
    items, warnings, all_segments, ignored = [], [], [], []
    last_entry = None
    for sentence in split_sentences(cleaned):
        segments, header = segment_sentence(sentence)
        ignored.extend(header)
        for seg in segments:
            entry = parse_segment(seg)
            if entry is None:
                continue
            all_segments.append(seg)
            if not entry["phrase"]:
                # quantity without an item: "rice, around 2 kg" -> attach to previous
                if (last_entry is not None and last_entry["quantity_status"] == "missing"
                        and entry["quantity"] is not None):
                    last_entry.update(quantity=entry["quantity"], unit=entry["unit"],
                                      unit_inferred=entry["unit_inferred"],
                                      quantity_text=entry["quantity_text"],
                                      quantity_status="specified")
                    last_entry["original_phrase"] += f", {entry['original_phrase']}"
                    items[-1] = build_item(last_entry)
                else:
                    warnings.append(f"Missing item name: '{entry['original_phrase']}' "
                                    "has a quantity but no item.")
                continue
            if (not re.search(r"[a-z]{2,}", entry["phrase"])
                    or set(entry["phrase"].split()) <= NON_ITEM_WORDS):
                warnings.append(f"Unrecognized phrase skipped: '{entry['original_phrase']}'.")
                continue
            items.append(build_item(entry))
            last_entry = entry
    for it in items:
        if it["quantity_status"] == "missing":
            warnings.append(f"Quantity was not specified for {it['item']}.")
        if it["category"] == "Other" and it["status"] == "Extracted":
            warnings.append(f"'{it['item']}' is not in the category dictionary, so it is listed as Other.")
        for n in it["notes"]:
            warnings.append(f"{it['display_name']}: {n}")
    return {"items": items, "warnings": warnings, "segments": all_segments,
            "ignored_phrases": ignored}
