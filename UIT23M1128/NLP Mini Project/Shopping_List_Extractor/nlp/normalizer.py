"""Item normalisation: lemmatisation + a small controlled exception list."""
from nlp.preprocessing import lemmatize_word

IRREGULAR = {
    "tomatoes": "tomato", "potatoes": "potato", "mangoes": "mango",
    "loaves": "loaf", "leaves": "leaf", "berries": "berry", "cherries": "cherry",
    "knives": "knife", "peaches": "peach", "radishes": "radish",
    "sandwiches": "sandwich", "dishes": "dish", "boxes": "box",
}
# Words that should keep their (apparently plural) form
INVARIANT = {"hummus", "couscous", "asparagus", "molasses", "oats", "cornflakes",
             "swiss", "citrus", "lychee", "series"}


def normalize_word(word):
    w = word.lower().strip()
    if w in INVARIANT:
        return w
    if w in IRREGULAR:
        return IRREGULAR[w]
    return lemmatize_word(w)


def normalize_item(phrase):
    """'Tomatoes' -> 'tomato', 'Floor Cleaners' -> 'floor cleaner'.

    Only the head noun (last word) is lemmatised, so 'baked beans' -> 'baked bean'.
    """
    words = phrase.lower().split()
    if not words:
        return ""
    words[-1] = normalize_word(words[-1])
    return " ".join(words)


def display_name(normalized):
    return normalized.capitalize() if normalized else ""
