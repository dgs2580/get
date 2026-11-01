"""Text preprocessing with NLTK: cleaning, sentence segmentation, tokenisation,
stopword handling and lemmatisation.

NLTK resources are downloaded automatically on first use. If a download is
impossible (offline machine) small regex fallbacks keep the app running.
"""
import re

import nltk

from nlp.quantity_extractor import FRACTION_WORDS, NUMBER_WORDS
from nlp.unit_extractor import UNIT_ALIASES

_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}
_ready = False
_lemmatizer = None
_stopwords = None

# Words that are important for extraction and must never be stripped
PRESERVE = set(UNIT_ALIASES) | set(NUMBER_WORDS) | set(FRACTION_WORDS) | {"dozen"}

_FALLBACK_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "to", "i", "me", "my", "we", "our", "you",
    "for", "in", "on", "at", "is", "are", "be", "it", "that", "this", "some", "any",
    "with", "from", "as", "so", "do", "does", "can", "will", "would", "could",
}


def ensure_nltk_resources():
    """Download missing NLTK resources once per process."""
    global _ready
    if _ready:
        return
    for pkg, path in _RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:  # offline - fallbacks will be used
                pass
    _ready = True


def _get_stopwords():
    global _stopwords
    if _stopwords is None:
        ensure_nltk_resources()
        try:
            from nltk.corpus import stopwords
            _stopwords = set(stopwords.words("english"))
        except Exception:
            _stopwords = set(_FALLBACK_STOPWORDS)
    return _stopwords


def _get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        ensure_nltk_resources()
        try:
            from nltk.stem import WordNetLemmatizer
            lem = WordNetLemmatizer()
            lem.lemmatize("tests")  # raises LookupError if WordNet is missing
            _lemmatizer = lem
        except Exception:
            _lemmatizer = False
    return _lemmatizer


def lemmatize_word(word):
    lem = _get_lemmatizer()
    if lem:
        return lem.lemmatize(word.lower(), pos="n")
    w = word.lower()  # crude fallback
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("oes"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


# --- cleaning ---------------------------------------------------------------
def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text):
    """Normalise quotes/whitespace, split '500g' -> '500 g', drop 1,000 separators."""
    t = str(text or "")
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", t)
    return normalize_whitespace(t)


def split_sentences(text):
    ensure_nltk_resources()
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]


def tokenize(text):
    """NLTK word tokenisation (lower-cased); regex fallback if unavailable."""
    ensure_nltk_resources()
    try:
        from nltk.tokenize import word_tokenize
        toks = word_tokenize(text.lower())
    except Exception:
        toks = re.findall(r"\d+(?:[./]\d+)?|[a-z]+(?:'[a-z]+)?|[^\sa-z\d]", text.lower())
    return [t for t in toks if t.strip()]


def is_word_token(token):
    return bool(re.search(r"[a-z0-9]", token))


def remove_stopwords(tokens):
    """Remove stopwords but keep numbers, units, quantity words and punctuation-free items."""
    sw = _get_stopwords()
    kept = []
    for t in tokens:
        if not is_word_token(t):
            continue
        if any(ch.isdigit() for ch in t) or t in PRESERVE or t not in sw:
            kept.append(t)
    return kept


def lemmatize_tokens(tokens):
    return [t if any(ch.isdigit() for ch in t) or t in PRESERVE else lemmatize_word(t)
            for t in tokens]


def preprocess(text):
    """Full preprocessing pipeline, returned step by step for explainability."""
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    tokens = tokenize(cleaned)
    word_tokens = [t for t in tokens if is_word_token(t)]
    no_stop = remove_stopwords(tokens)
    return {
        "original_text": text,
        "cleaned_text": cleaned.lower(),
        "sentences": sentences,
        "tokens": word_tokens,
        "tokens_without_stopwords": no_stop,
        "lemmas": lemmatize_tokens(no_stop),
    }
