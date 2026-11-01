"""Orchestrates the whole NLP pipeline and builds the structured JSON result."""
import re
from collections import Counter
from datetime import datetime

from categories.category_data import CATEGORIES
from nlp.item_extractor import build_item, extract_items
from nlp.keyword_extractor import extract_keywords
from nlp.normalizer import normalize_item
from nlp.preprocessing import preprocess
from nlp.unit_extractor import convert, display_unit, format_number, supported_units

MIN_CHARS = 3
MAX_CHARS = 5000


class ValidationError(ValueError):
    """Raised for input that cannot be processed (shown to the user as-is)."""


def validate_text(text):
    if text is None or not isinstance(text, str):
        raise ValidationError("Input text is missing. Please enter a shopping request.")
    t = text.strip()
    if not t:
        raise ValidationError("Input is empty. Please type your shopping request.")
    if len(t) < MIN_CHARS:
        raise ValidationError(f"Input is too short (minimum {MIN_CHARS} characters).")
    if len(t) > MAX_CHARS:
        raise ValidationError(f"Input is too long ({len(t)} characters). Maximum is {MAX_CHARS}.")
    if not re.search(r"[A-Za-z]{2,}", t):
        raise ValidationError("The text does not contain any recognizable words.")
    return t


# ---------------------------------------------------------------- duplicates
def merge_duplicates(items):
    """Merge repeated items when quantities are compatible (same unit or convertible).

    Returns (items, notes, duplicate_count). Incompatible entries stay separate
    and are marked 'Duplicate'. Nothing is merged silently - notes explain it.
    """
    counts = Counter(i["item"] for i in items)
    duplicate_count = sum(c - 1 for c in counts.values() if c > 1)
    result, notes = [], []
    seen = {}
    for it in items:
        key = it["item"]
        if counts[key] == 1:
            result.append(it)
            continue
        merged = False
        for base in seen.get(key, []):
            if base["quantity"] is None and it["quantity"] is None \
                    and base["quantity_status"] == it["quantity_status"] == "missing":
                base["status"] = "Merged"
                base.setdefault("merged_from", [base["original_phrase"]]).append(it["original_phrase"])
                notes.append(f"{base['display_name']}: repeated mention combined (quantity not specified).")
                merged = True
            elif base["quantity"] is not None and it["quantity"] is not None:
                conv = convert(it["quantity"], it["unit"], base["unit"])
                if conv is not None:
                    before = f"{base['quantity_display']} + {it['quantity_display']}"
                    total = round(base["quantity"] + conv, 3)
                    unit = base["unit"]
                    if unit == "g" and total >= 1000:
                        total, unit = total / 1000, "kg"
                    elif unit == "ml" and total >= 1000:
                        total, unit = total / 1000, "litre"
                    base["quantity"], base["unit"] = round(total, 3), unit
                    base["unit_display"] = display_unit(unit, base["quantity"])
                    base["quantity_display"] = f"{format_number(base['quantity'])} {base['unit_display']}"
                    base["status"] = "Merged"
                    base.setdefault("merged_from", [base["original_phrase"]]).append(it["original_phrase"])
                    notes.append(f"{base['display_name']}: duplicates combined ({before} = {base['quantity_display']}).")
                    merged = True
            if merged:
                break
        if not merged:
            if key in seen:
                it["status"] = "Duplicate"
                notes.append(f"{it['display_name']} appears more than once with incompatible or "
                             "missing quantities, so the entries were kept separate.")
            seen.setdefault(key, []).append(it)
            result.append(it)
    # first occurrence of any duplicated-but-unmerged item is flagged too
    for key, group in seen.items():
        if len(group) > 1 and group[0]["status"] == "Extracted":
            group[0]["status"] = "Duplicate"
    return result, notes, duplicate_count


# ---------------------------------------------------------------- statistics
def build_statistics(items, duplicate_count=0):
    total = len(items)
    specified = sum(1 for i in items if i["quantity_status"] == "specified")
    invalid = sum(1 for i in items if i["quantity_status"] == "invalid")
    cats = Counter(i["category"] for i in items)
    most_common = cats.most_common(1)[0][0] if cats else "—"
    return {
        "total_items": total,
        "specified_quantities": specified,
        "missing_quantities": total - specified,
        "invalid_quantities": invalid,
        "categories": len(cats),
        "most_common_category": most_common,
        "duplicate_items": duplicate_count,
        "merged_items": sum(1 for i in items if i.get("status") == "Merged"),
        "category_counts": dict(cats.most_common()),
    }


def category_summary(items):
    grouped = {}
    for cat in CATEGORIES:
        rows = [i for i in items if i["category"] == cat]
        if rows:
            grouped[cat] = [{"item": r["display_name"], "quantity_display": r["quantity_display"]}
                            for r in rows]
    return grouped


def generate_summary(items):
    if not items:
        return "No shopping items were extracted."
    parts = []
    for i in items:
        if i["quantity_status"] == "specified":
            parts.append(f"{i['display_name']}: {i['quantity_display']}.")
        else:
            parts.append(f"{i['display_name']}: quantity not specified.")
    word = "item" if len(items) == 1 else "items"
    return f"Your shopping list contains {len(items)} {word}. " + " ".join(parts)


# ---------------------------------------------------------------- pipelines
def _finish(items, extra_notes, dup_count, warnings, keywords, merge):
    stats = build_statistics(items, dup_count)
    return {
        "items": items, "keywords": keywords, "statistics": stats,
        "category_summary": category_summary(items), "summary": generate_summary(items),
        "warnings": warnings, "merge_notes": extra_notes,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def analyze_text(text, merge=True):
    """Run the full NLP pipeline on a natural-language shopping request."""
    text = validate_text(text)
    pre = preprocess(text)
    extraction = extract_items(text)
    raw_items = extraction["items"]
    if merge:
        items, notes, dups = merge_duplicates([dict(i) for i in raw_items])
    else:
        counts = Counter(i["item"] for i in raw_items)
        items, notes, dups = raw_items, [], sum(c - 1 for c in counts.values() if c > 1)
    docs = [i["item"] for i in raw_items] or [" ".join(pre["lemmas"])]
    keywords = extract_keywords(docs)
    warnings = list(extraction["warnings"])
    if not items:
        warnings.append("No shopping items could be recognized in the text.")
    if dups and not merge:
        warnings.append(f"{dups} duplicate item(s) detected and kept as separate entries.")
    result = _finish(items, notes, dups, warnings, keywords, merge)
    result["explainability"] = {
        **pre,
        "segments": extraction["segments"],
        "ignored_phrases": extraction["ignored_phrases"],
        "extraction_trace": [{
            "original_phrase": i["original_phrase"], "item": i["item"],
            "quantity": i["quantity_display"], "quantity_matched_text": i["quantity_text"] or "—",
            "category": i["category"], "category_method": i["category_method"],
            "confidence": i["confidence"]} for i in raw_items],
        "supported_units": supported_units(),
    }
    return result


def analyze_items(raw_rows, merge=True):
    """Analyse already-structured rows (manual input / CSV).

    raw_rows: list of dicts with item, quantity, unit, category (all optional but item).
    Invalid values are flagged, never silently fixed.
    """
    from nlp.category_classifier import classify_item, valid_category
    from nlp.unit_extractor import canonical_unit

    items, warnings = [], []
    for n, row in enumerate(raw_rows, start=1):
        name = str(row.get("item") or "").strip()
        if not name or name.lower() == "nan":
            warnings.append(f"Row {n}: missing item name - skipped.")
            continue
        notes, status = [], "Extracted"
        qty_raw = row.get("quantity")
        qty, qstatus = None, "missing"
        if qty_raw is not None and str(qty_raw).strip() not in ("", "nan", "None"):
            try:
                qty = float(str(qty_raw).replace(",", ""))
                if qty <= 0:
                    notes.append("Quantity must be greater than zero; value ignored.")
                    qty, qstatus, status = None, "invalid", "Invalid quantity"
                else:
                    qstatus = "specified"
            except ValueError:
                notes.append(f"'{qty_raw}' is not a valid quantity.")
                qstatus, status = "invalid", "Invalid quantity"
        unit_raw = str(row.get("unit") or "").strip()
        unit = None
        if unit_raw and unit_raw.lower() != "nan":
            unit = canonical_unit(unit_raw)
            if unit is None:
                notes.append(f"Unit '{unit_raw}' is not supported.")
                status = "Unsupported unit" if status == "Extracted" else status
        if unit == "dozen" and qty is not None:
            qty, unit = qty * 12, "piece"
        normalized = normalize_item(name.lower())
        cat_in = row.get("category")
        cat = valid_category(cat_in) if cat_in and str(cat_in).lower() != "nan" else None
        if cat:
            method, conf = "provided", 1.0
        else:
            c = classify_item(normalized)
            cat, method, conf = c["category"], c["method"], c["confidence"]
            if cat_in and str(cat_in).lower() != "nan":
                notes.append(f"Category '{cat_in}' is not a known category; it was re-classified.")
        entry = {"original_phrase": name, "phrase": normalized, "quantity": qty, "unit": unit,
                 "unit_inferred": False, "quantity_text": "", "quantity_status": qstatus,
                 "status": status, "notes": notes}
        it = build_item(entry)
        it["category"], it["category_method"] = cat, method
        items.append(it)
        warnings += [f"{it['display_name']}: {n_}" for n_ in notes]
        if qstatus == "missing":
            warnings.append(f"Quantity was not specified for {it['item']}.")
    if merge:
        items, notes, dups = merge_duplicates(items)
    else:
        counts = Counter(i["item"] for i in items)
        notes, dups = [], sum(c - 1 for c in counts.values() if c > 1)
    keywords = extract_keywords([i["item"] for i in items])
    return _finish(items, notes, dups, warnings, keywords, merge)


# ---------------------------------------------------------------- comparison
def compare_lists(items_a, items_b):
    """Compare two item lists by normalised item name."""
    def index(items):
        d = {}
        for i in items:
            d.setdefault(i["item"], []).append(i)
        return d
    a, b = index(items_a), index(items_b)
    common = [k for k in a if k in b]
    only_a = [k for k in a if k not in b]
    only_b = [k for k in b if k not in a]

    def label(group):
        return ", ".join(x["quantity_display"] for x in group)

    return {
        "common": [{"item": a[k][0]["display_name"], "list_a": label(a[k]), "list_b": label(b[k]),
                    "same_quantity": label(a[k]) == label(b[k])} for k in common],
        "only_in_a": [{"item": a[k][0]["display_name"], "quantity": label(a[k])} for k in only_a],
        "only_in_b": [{"item": b[k][0]["display_name"], "quantity": label(b[k])} for k in only_b],
        "counts": {"list_a": len(a), "list_b": len(b), "common": len(common),
                   "only_a": len(only_a), "only_b": len(only_b)},
    }
