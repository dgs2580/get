import pytest

from nlp.category_classifier import classify_item, valid_category


@pytest.mark.parametrize("item,cat", [
    ("apple", "Fruits"), ("milk", "Dairy"), ("rice", "Grains"), ("bread", "Bakery"),
    ("soap", "Personal Care"), ("detergent", "Cleaning"), ("tomato", "Vegetables"),
    ("floor cleaner", "Cleaning"), ("basmati rice", "Grains"), ("orange juice", "Beverages"),
    ("pen", "Stationery"), ("chicken", "Meat"), ("salmon", "Seafood")])
def test_known_categories(item, cat):
    assert classify_item(item)["category"] == cat


@pytest.mark.parametrize("item", ["dragon fruit", "rice cooker", "zzzz", ""])
def test_unknown_is_other_not_guessed(item):
    r = classify_item(item)
    assert r["category"] == "Other" and r["confidence"] == 0.0


def test_buy_apples_pipeline():
    from nlp.text_analyzer import analyze_text
    assert analyze_text("Buy apples.")["items"][0]["category"] == "Fruits"


def test_valid_category_case_insensitive():
    assert valid_category("personal care") == "Personal Care"
    assert valid_category("nonsense") is None
