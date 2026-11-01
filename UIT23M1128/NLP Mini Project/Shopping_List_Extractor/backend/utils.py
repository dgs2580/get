"""Backend helpers: JSON responses, CSV parsing/normalisation, report builders."""
import io
from datetime import datetime

import pandas as pd
from flask import jsonify

from nlp.unit_extractor import format_number

COLUMN_ALIASES = {
    "item": {"item", "items", "item_name", "itemname", "product", "product_name", "name"},
    "quantity": {"quantity", "qty", "amount", "count"},
    "unit": {"unit", "units", "uom"},
    "category": {"category", "category_name", "type", "group"},
}
MAX_CSV_ROWS = 5000


def ok(payload=None, **extra):
    body = {"status": "success"}
    body.update(payload or {})
    body.update(extra)
    return jsonify(body)


def fail(message, code=400, error_code="bad_request"):
    return jsonify({"status": "error", "message": message, "code": error_code}), code


def normalize_columns(df):
    """Rename aliased columns to item/quantity/unit/category. Returns (df, missing_required)."""
    mapping = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        for canonical, aliases in COLUMN_ALIASES.items():
            if key in aliases and canonical not in mapping.values():
                mapping[col] = canonical
    df = df.rename(columns=mapping)
    missing = [] if "item" in df.columns else ["item"]
    return df, missing


def read_csv_file(file_storage):
    """Parse an uploaded CSV into a normalised DataFrame; raises ValueError with friendly text."""
    raw = file_storage.read()
    if not raw or not raw.strip():
        raise ValueError("The uploaded CSV file is empty.")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.StringIO(raw.decode(enc)))
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded CSV file has no data.")
        except pd.errors.ParserError:
            raise ValueError("The CSV file is malformed and could not be parsed.")
    else:
        raise ValueError("The CSV file encoding is not supported.")
    if df.empty:
        raise ValueError("The CSV file has headers but no rows.")
    if len(df) > MAX_CSV_ROWS:
        raise ValueError(f"The CSV has {len(df)} rows; the maximum is {MAX_CSV_ROWS}.")
    df, missing = normalize_columns(df)
    if missing:
        raise ValueError("Missing required column: 'item' (accepted names: item, item_name, product).")
    return df


def df_to_rows(df):
    rows = []
    for _, r in df.iterrows():
        rows.append({k: (None if pd.isna(r.get(k)) else r.get(k))
                     for k in ("item", "quantity", "unit", "category") if k in df.columns})
    return rows


def items_to_csv(items):
    df = pd.DataFrame([{
        "item": i["display_name"],
        "quantity": "" if i["quantity"] is None else format_number(i["quantity"]),
        "unit": i["unit"] or "",
        "category": i["category"],
        "status": i["status"],
    } for i in items])
    return df.to_csv(index=False)


def build_txt_report(items, statistics, keywords, original_text=None):
    lines = ["# SHOPPING LIST EXTRACTOR", "",
             f"Analysis date/time: {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    if original_text:
        lines += ["Original request:", original_text.strip(), ""]
    lines += ["Extracted Shopping List", ""]
    for n, i in enumerate(items, 1):
        lines.append(f"{n}. {i['display_name']} — {i['quantity_display']} — {i['category']}")
    if not items:
        lines.append("(no items)")
    s = statistics or {}
    lines += ["", "Statistics:", "",
              f"Total Items: {s.get('total_items', 0)}",
              f"Specified Quantities: {s.get('specified_quantities', 0)}",
              f"Missing Quantities: {s.get('missing_quantities', 0)}",
              f"Categories: {s.get('categories', 0)}",
              f"Most Common Category: {s.get('most_common_category', '—')}",
              f"Duplicate Items: {s.get('duplicate_items', 0)}",
              "", "Top NLP Keywords (relative keyword importance):", ""]
    for k in keywords or []:
        lines.append(f"{k['keyword']}  ({k.get('relative_importance', 0):.2f})")
    return "\n".join(lines) + "\n"
