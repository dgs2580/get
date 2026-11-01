"""REST API routes (JSON in / JSON out). No stack traces are ever returned."""
import logging

from flask import Blueprint, request

from backend.utils import (build_txt_report, df_to_rows, fail, items_to_csv, ok,
                           read_csv_file)
from nlp.category_classifier import classify_item
from nlp.normalizer import normalize_item
from nlp.text_analyzer import (ValidationError, analyze_items, analyze_text,
                               compare_lists, validate_text)

api = Blueprint("api", __name__, url_prefix="/api")
log = logging.getLogger(__name__)


def _json():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def _list_from(payload, label):
    """A list can be given as natural-language text or as structured items."""
    if isinstance(payload.get("items"), list):
        return analyze_items(payload["items"], merge=True)["items"]
    text = payload.get(label if label in payload else "text")
    return analyze_text(text, merge=True)["items"]


@api.get("/health")
def health():
    return ok({"service": "shopping-list-extractor", "message": "Backend is running"})


@api.post("/extract")
def extract():
    data = _json()
    if data is None:
        return fail("Request body must be JSON.", 400, "invalid_json")
    try:
        r = analyze_text(data.get("text"), merge=bool(data.get("merge_duplicates", True)))
    except ValidationError as e:
        return fail(str(e), 400, "validation_error")
    r.pop("explainability", None)
    return ok(r)


@api.post("/analyze")
def analyze():
    """Full analysis incl. explainability. Accepts `text` or structured `items`."""
    data = _json()
    if data is None:
        return fail("Request body must be JSON.", 400, "invalid_json")
    merge = bool(data.get("merge_duplicates", True))
    try:
        if isinstance(data.get("items"), list):
            if not data["items"]:
                return fail("The item list is empty.", 400, "validation_error")
            return ok(analyze_items(data["items"], merge=merge))
        return ok(analyze_text(data.get("text"), merge=merge))
    except ValidationError as e:
        return fail(str(e), 400, "validation_error")


@api.post("/compare")
def compare():
    data = _json()
    if data is None:
        return fail("Request body must be JSON.", 400, "invalid_json")
    try:
        a = _list_from({"text": data.get("list_a"), "items": data.get("items_a")}, "text")
        b = _list_from({"text": data.get("list_b"), "items": data.get("items_b")}, "text")
    except ValidationError as e:
        return fail(f"Both lists need valid text. {e}", 400, "validation_error")
    return ok({"comparison": compare_lists(a, b)})


@api.post("/categorize")
def categorize():
    data = _json()
    if data is None or not isinstance(data.get("items"), list) or not data["items"]:
        return fail("Provide a non-empty 'items' list of item names.", 400, "validation_error")
    out = []
    for name in data["items"][:200]:
        name = str(name).strip()
        if not name:
            continue
        norm = normalize_item(name.lower())
        c = classify_item(norm)
        out.append({"input": name, "item": norm, **c})
    return ok({"results": out})


@api.post("/upload")
def upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return fail("No file uploaded. Please choose a .csv file.", 400, "no_file")
    if not f.filename.lower().endswith(".csv"):
        return fail("Only .csv files are accepted.", 400, "bad_file_type")
    try:
        df = read_csv_file(f)
    except ValueError as e:
        return fail(str(e), 400, "invalid_csv")
    r = analyze_items(df_to_rows(df), merge=False)
    r["preview"] = df.head(20).fillna("").astype(str).to_dict(orient="records")
    r["columns"] = [str(c) for c in df.columns]
    r["rows"] = int(len(df))
    return ok(r)


@api.post("/report")
def report():
    data = _json()
    if data is None or not isinstance(data.get("items"), list):
        return fail("Provide the extracted 'items' list.", 400, "validation_error")
    items = data["items"]
    return ok({"txt": build_txt_report(items, data.get("statistics"), data.get("keywords"),
                                       data.get("original_text")),
               "csv": items_to_csv(items)})
