import pytest

from nlp.keyword_extractor import extract_keywords
from nlp.normalizer import normalize_item
from nlp.preprocessing import preprocess, remove_stopwords, tokenize
from nlp.quantity_extractor import find_quantity, parse_number
from nlp.text_analyzer import ValidationError, validate_text
from nlp.unit_extractor import canonical_unit, display_unit


def test_tokenization_preserves_numbers_units_items():
    toks = preprocess("2 kg rice")["tokens"]
    assert toks == ["2", "kg", "rice"]


def test_stopword_removal_keeps_quantity_words_and_units():
    kept = remove_stopwords(tokenize("please buy half a kilo of rice and 2 kg flour"))
    for w in ("half", "kilo", "rice", "2", "kg", "flour"):
        assert w in kept
    assert "of" not in kept and "and" not in kept


def test_clean_text_splits_attached_units():
    assert "500 g" in preprocess("Buy 500g sugar")["cleaned_text"]


def test_sentence_segmentation():
    assert len(preprocess("Buy rice. Add 2 litres of milk.")["sentences"]) == 2


@pytest.mark.parametrize("a,b", [("tomatoes", "tomato"), ("apples", "apple"),
                                 ("potatoes", "potato"), ("biscuits", "biscuit"),
                                 ("eggs", "egg"), ("TOMATOES", "tomato")])
def test_item_normalization(a, b):
    assert normalize_item(a) == b


def test_unit_normalization():
    assert canonical_unit("Kilograms") == "kg"
    assert canonical_unit("liters") == "litre"
    assert display_unit("litre", 2) == "litres"
    assert display_unit("kg", 2) == "kg"


def test_number_words():
    assert parse_number("five") == 5
    assert parse_number("2.5") == 2.5
    assert parse_number("rice") is None


def test_half_a_kilo():
    q = find_quantity("half a kilo of rice".split())
    assert (q["quantity"], q["unit"]) == (0.5, "kg")


def test_tfidf_keywords_come_from_input():
    kws = extract_keywords(["basmati rice", "milk", "apple"])
    words = {k["keyword"] for k in kws}
    assert words == {"basmati", "rice", "milk", "apple"}
    assert all(0 <= k["relative_importance"] <= 1 for k in kws)


def test_validation_errors():
    for bad in ("", "   ", "ab", None, "12345", "x" * 6000):
        with pytest.raises(ValidationError):
            validate_text(bad)
