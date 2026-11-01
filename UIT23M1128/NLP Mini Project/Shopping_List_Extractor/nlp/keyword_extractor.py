"""TF-IDF keyword analysis with scikit-learn.

TF  = share of the request's terms that a term makes up
IDF = inverse document frequency learned by sklearn's TfidfVectorizer on a
      background corpus (sample shopping lists) + the request's own phrases
Score = TF x IDF. It is a *relative importance* value, not a probability.
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(ROOT, "data", "sample_shopping_lists.csv")

_FALLBACK_CORPUS = ["rice", "milk", "apple", "bread", "tomato", "soap", "egg",
                    "sugar", "butter", "banana", "potato", "onion", "detergent"]
_background = None


def background_corpus():
    """Each sample item becomes one small 'document'."""
    global _background
    if _background is None:
        docs = []
        try:
            df = pd.read_csv(SAMPLE_CSV)
            docs = [str(x).lower().strip() for x in df["item"].dropna()]
        except Exception:
            pass
        _background = docs or list(_FALLBACK_CORPUS)
    return _background


def extract_keywords(documents, top_n=10):
    """documents: list of strings (one per extracted phrase). Returns ranked terms."""
    docs = [d.strip().lower() for d in documents if d and d.strip()]
    if not docs:
        return []
    corpus = background_corpus() + docs
    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                          norm=None, smooth_idf=True)
    vec.fit(corpus)
    vocab = vec.vocabulary_
    idf = vec.idf_

    terms = [t for d in docs for t in d.split()]
    total = len(terms)
    scored = []
    for term in dict.fromkeys(terms):  # unique, order preserved
        tf = terms.count(term) / total
        term_idf = float(idf[vocab[term]])
        scored.append({"keyword": term, "tf": round(tf, 4),
                       "idf": round(term_idf, 4), "tfidf": round(tf * term_idf, 4)})
    scored.sort(key=lambda r: (-r["tfidf"], r["keyword"]))
    top = scored[:top_n]
    best = top[0]["tfidf"] if top else 1
    for row in top:
        row["relative_importance"] = round(row["tfidf"] / best, 3) if best else 0.0
    return top
