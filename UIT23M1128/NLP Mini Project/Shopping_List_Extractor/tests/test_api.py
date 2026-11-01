import io

import pytest

from backend.app import create_app


@pytest.fixture()
def client():
    return create_app().test_client()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json["status"] == "success"


def test_extract(client):
    r = client.post("/api/extract", json={"text": "Buy 2 kg rice and 1 litre milk."})
    assert r.status_code == 200
    items = {i["item"]: i for i in r.json["items"]}
    assert items["rice"]["quantity"] == 2 and items["rice"]["unit"] == "kg"
    assert items["milk"]["category"] == "Dairy"
    assert "keywords" in r.json and "statistics" in r.json


def test_extract_empty_is_validation_error(client):
    r = client.post("/api/extract", json={"text": "  "})
    assert r.status_code == 400 and r.json["status"] == "error"


def test_extract_bad_body(client):
    r = client.post("/api/extract", data="not json", content_type="text/plain")
    assert r.status_code == 400


def test_analyze_has_explainability(client):
    r = client.post("/api/analyze", json={"text": "Please buy 2 kg rice and 1 litre milk."})
    assert r.status_code == 200
    ex = r.json["explainability"]
    assert "rice" in ex["tokens"] and ex["cleaned_text"]


def test_analyze_structured_items(client):
    r = client.post("/api/analyze", json={"items": [{"item": "Apples", "quantity": "3", "unit": "kg"},
                                                    {"item": "soap", "quantity": "-1"}]})
    assert r.status_code == 200
    s = r.json["statistics"]
    assert s["total_items"] == 2 and s["invalid_quantities"] == 1


def test_compare(client):
    r = client.post("/api/compare", json={"list_a": "Rice, milk and apples",
                                          "list_b": "Rice, milk, bread and eggs"})
    c = r.json["comparison"]
    assert [x["item"] for x in c["common"]] == ["Rice", "Milk"]
    assert [x["item"] for x in c["only_in_a"]] == ["Apple"]
    assert [x["item"] for x in c["only_in_b"]] == ["Bread", "Egg"]


def test_categorize(client):
    r = client.post("/api/categorize", json={"items": ["apples", "detergent", "dragon fruit"]})
    cats = [x["category"] for x in r.json["results"]]
    assert cats == ["Fruits", "Cleaning", "Other"]


def test_upload_csv(client):
    csv = b"Product,qty,units\nrice,2,kg\nmilk,,litre\nrice,1,kg\n"
    r = client.post("/api/upload", data={"file": (io.BytesIO(csv), "x.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    s = r.json["statistics"]
    assert s["total_items"] == 3 and s["missing_quantities"] == 1 and s["duplicate_items"] == 1


def test_upload_missing_column_and_wrong_type(client):
    r = client.post("/api/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "item" in r.json["message"]
    r = client.post("/api/upload", data={"file": (io.BytesIO(b"x"), "x.txt")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_report(client):
    ex = client.post("/api/extract", json={"text": "Buy 2 kg rice."}).json
    r = client.post("/api/report", json={"items": ex["items"], "statistics": ex["statistics"],
                                         "keywords": ex["keywords"]})
    assert "Rice — 2 kg — Grains" in r.json["txt"] and "Rice,2,kg,Grains" in r.json["csv"]


def test_unknown_route_is_json(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json["status"] == "error"
