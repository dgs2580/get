import pytest

from nlp.text_analyzer import analyze_text


def items_of(text, **kw):
    return analyze_text(text, **kw)["items"]


def as_dict(items):
    return {i["item"]: i for i in items}


def test_basic_item_extraction():
    d = as_dict(items_of("Buy rice and milk."))
    assert set(d) == {"rice", "milk"}


def test_quantity_extraction():
    i = items_of("Buy 2 kg rice.")[0]
    assert (i["item"], i["quantity"], i["unit"]) == ("rice", 2, "kg")


def test_word_number_extraction():
    i = items_of("Buy five apples.")[0]
    assert i["item"] == "apple" and i["quantity"] == 5


def test_multiple_items_and_correct_association():
    d = as_dict(items_of("Buy 2 kg rice, 1 litre milk and 6 apples."))
    assert len(d) == 3
    assert (d["rice"]["quantity"], d["rice"]["unit"]) == (2, "kg")
    assert (d["milk"]["quantity"], d["milk"]["unit"]) == (1, "litre")
    assert (d["apple"]["quantity"], d["apple"]["unit"]) == (6, "piece")


def test_missing_quantity_is_not_assumed():
    i = items_of("Buy tomatoes.")[0]
    assert i["quantity"] is None and i["quantity_display"] == "Not specified"
    assert i["quantity_status"] == "missing"


def test_no_fabricated_items_for_filler_words():
    names = {i["item"] for i in items_of("Please buy some tomatoes.")}
    assert names == {"tomato"}


@pytest.mark.parametrize("text", ["2 kg rice", "Please buy two kilograms of rice.",
                                  "I need some rice, around 2 kg.", "Add rice 2 kilograms to my list."])
def test_free_form_variations(text):
    i = items_of(text)[0]
    assert (i["item"], i["quantity"], i["unit"]) == ("rice", 2, "kg")


def test_dozen_and_half():
    d = as_dict(items_of("Please buy one dozen eggs, 2 kg rice and half a kilo of onions."))
    assert d["egg"]["quantity"] == 12
    assert (d["onion"]["quantity"], d["onion"]["unit"]) == (0.5, "kg")


def test_header_phrase_not_an_item():
    names = {i["item"] for i in items_of("Get some vegetables like tomatoes, onions and potatoes.")}
    assert names == {"tomato", "onion", "potato"}
    names = {i["item"] for i in items_of("Buy household items: detergent, soap and toothpaste.")}
    assert names == {"detergent", "soap", "toothpaste"}


def test_duplicate_merge_same_unit():
    r = analyze_text("Buy 2 kg rice, and also add 1 kg rice.")
    assert len(r["items"]) == 1
    assert r["items"][0]["quantity"] == 3 and r["items"][0]["status"] == "Merged"
    assert r["statistics"]["duplicate_items"] == 1


def test_duplicate_merge_with_unit_conversion():
    r = analyze_text("Buy 500 g sugar and 1 kg sugar")
    assert r["items"][0]["quantity_display"] == "1.5 kg"


def test_duplicate_incompatible_not_merged():
    r = analyze_text("Buy rice and 2 kg rice.")
    assert len(r["items"]) == 2
    assert all(i["status"] == "Duplicate" for i in r["items"])


def test_duplicates_kept_when_merge_disabled():
    assert len(analyze_text("Buy 2 kg rice and 1 kg rice", merge=False)["items"]) == 2


@pytest.mark.parametrize("text,status", [("Buy -2 kg rice", "Invalid quantity"),
                                         ("Buy 0 apples", "Invalid quantity"),
                                         ("Buy 1.2.3 kg sugar", "Invalid quantity"),
                                         ("Buy 3 foo of milk", "Unsupported unit")])
def test_invalid_quantities_are_flagged_not_assumed(text, status):
    i = items_of(text)[0]
    assert i["status"] == status and i["quantity"] is None


def test_unknown_item_kept_as_other():
    i = items_of("Buy dragon fruit.")[0]
    assert i["item"] == "dragon fruit" and i["category"] == "Other"


def test_full_example_statistics():
    r = analyze_text("Please buy 2 kg rice, one litre milk, six apples, 3 packets of biscuits, "
                     "500 grams cheese and some tomatoes.")
    s = r["statistics"]
    assert (s["total_items"], s["specified_quantities"], s["missing_quantities"], s["categories"]) == (6, 5, 1, 5)
    assert {k["keyword"] for k in r["keywords"]} >= {"rice", "milk", "apple", "biscuit", "cheese", "tomato"}
