"""Category classification: dictionary -> head-noun rule -> guarded TF-IDF similarity.

Never guesses: if there is no evidence the category is "Other".
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from categories.category_data import CATEGORIES, CATEGORY_KEYWORDS
from nlp.normalizer import normalize_word

SIMILARITY_THRESHOLD = 0.75  # char n-gram TF-IDF cosine needed to accept a fuzzy match


class CategoryClassifier:
    def __init__(self, keywords=None):
        keywords = keywords or CATEGORY_KEYWORDS
        self.lookup = {}
        for category, words in keywords.items():
            for w in words:
                self.lookup.setdefault(w.lower(), category)
        self.terms = sorted(self.lookup)
        # TF-IDF over character n-grams of the dictionary terms (handles typos)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform(self.terms)

    def _result(self, category, method, confidence, matched=None):
        return {"category": category, "method": method,
                "confidence": round(float(confidence), 2), "matched": matched}

    def classify(self, item):
        phrase = " ".join(item.lower().split())
        if not phrase:
            return self._result("Other", "unknown", 0.0)
        # 1. exact phrase
        if phrase in self.lookup:
            return self._result(self.lookup[phrase], "dictionary", 0.95, phrase)
        tokens = phrase.split()
        # 2. a multi-word dictionary entry contained in the phrase ("liquid floor cleaner")
        padded = f" {phrase} "
        for term in sorted(self.terms, key=len, reverse=True):
            if " " in term and f" {term} " in padded:
                return self._result(self.lookup[term], "dictionary-phrase", 0.9, term)
        # 3. head noun ("basmati rice" -> rice, "orange juice" -> juice)
        head = normalize_word(tokens[-1])
        if head in self.lookup:
            return self._result(self.lookup[head], "head-noun", 0.85, head)
        # 4. guarded fuzzy match on the head noun (misspellings like "tomatos")
        if len(head) >= 4:
            sims = cosine_similarity(self.vectorizer.transform([head]), self.matrix)[0]
            best = int(sims.argmax())
            term = self.terms[best]
            if sims[best] >= SIMILARITY_THRESHOLD and term[0] == head[0]:
                return self._result(self.lookup[term], "tfidf-similarity", sims[best], term)
        return self._result("Other", "unknown", 0.0)


_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = CategoryClassifier()
    return _classifier


def classify_item(item):
    return get_classifier().classify(item)


def valid_category(name):
    """Case-insensitive match against the configured categories, else None."""
    for c in CATEGORIES:
        if c.lower() == str(name).strip().lower():
            return c
    return None
