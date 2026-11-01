# SHOPPING LIST EXTRACTOR

An academic Natural Language Processing project that converts **natural-language shopping requests** into a **structured shopping list**.

> "I need 2 kg of rice, 1 litre milk, six apples, 3 packets of biscuits and some tomatoes."  
> → Rice · 2 kg · Grains | Milk · 1 litre · Dairy | Apple · 6 pieces · Fruits | Biscuit · 3 packets · Snacks | Tomato · *Not specified* · Vegetables

## Problem Statement
Shopping requests are written as free text. Turning them into a clean list requires finding the items, their quantities and units, and organising them — a classic **Information Extraction** task.

## Objectives
- Apply real NLP (NLTK + scikit-learn) to extract items, quantities, units and categories.
- Never invent data: missing values are reported as *Not specified* / *Other*.
- Provide a Flask REST backend and a Streamlit dashboard, with tests and documentation.

## Features
Natural-language input · quantity extraction (digits, decimals, number words, dozen/half/quarter) · unit normalisation · item normalisation (apples → apple) · 17 categories · duplicate detection/merging (with unit conversion) · TF-IDF keywords · NLP explainability page · statistics · Plotly charts · checklist (Pending/Purchased) · list comparison · CSV analysis · TXT/CSV downloads · validation and friendly errors.

## NLP Techniques Used
| Step | Technique | Library |
|---|---|---|
| 1 Input | Natural-language request | – |
| 2 Preprocessing | cleaning, sentence segmentation, tokenisation, stopword removal (numbers/units preserved), lemmatisation | **NLTK** |
| 3 Quantity extraction | regex + rules over tokens (numbers, number words, fractions, dozen/half) | rules |
| 4 Unit extraction | unit dictionary + canonical forms (kilograms → kg, liters → litre) | rules |
| 5 Item extraction | rule-based entity extraction (strips request phrases like "please buy", header phrases like "vegetables like …") | NLTK + rules |
| 6 Normalisation | WordNet lemmatiser + controlled exceptions | NLTK |
| 7 Category classification | dictionary → head-noun rule → guarded character-n-gram **TF-IDF similarity** → else *Other* | scikit-learn |
| 8 TF-IDF keywords | TF × IDF, IDF learned by `TfidfVectorizer` on a background corpus (the sample CSV) | scikit-learn |
| 9 Structured output | JSON → table, cards, categories, reports | pandas |

Quantities are attached only to the phrase they were written in, so `2 kg rice, 1 litre milk` can never be mixed up. TF-IDF values are shown as **Relative Keyword Importance**, not probabilities.

## System Architecture
```
USER → STREAMLIT UI → (HTTP REST) → FLASK BACKEND
                                     ├─ nlp/preprocessing.py       (NLTK)
                                     ├─ nlp/item_extractor.py      (entities)
                                     ├─ nlp/quantity_extractor.py / unit_extractor.py
                                     ├─ nlp/normalizer.py
                                     ├─ nlp/category_classifier.py (scikit-learn)
                                     └─ nlp/keyword_extractor.py   (scikit-learn TF-IDF)
                                  → STRUCTURED JSON → UI (list, charts, downloads)
```

## Project Structure
```
Shopping_List_Extractor/
├── frontend/app.py              Streamlit dashboard
├── backend/ app.py routes.py utils.py   Flask REST API
├── nlp/ preprocessing.py item_extractor.py quantity_extractor.py unit_extractor.py
│        normalizer.py keyword_extractor.py category_classifier.py text_analyzer.py
├── categories/category_data.py  category dictionary
├── data/sample_shopping_lists.csv
├── tests/ test_nlp.py test_extraction.py test_category.py test_api.py
├── reports/  exports/  screenshots/
├── .streamlit/config.toml  .env.example  .gitignore
└── requirements.txt  Procfile  runtime.txt
```

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```
NLTK data (punkt, punkt_tab, stopwords, wordnet, omw-1.4) downloads automatically on first run (internet needed once).

## Running Backend
```bash
python backend/app.py            # http://127.0.0.1:5000
```
## Running Frontend (second terminal)
```bash
streamlit run frontend/app.py    # http://localhost:8501
```
`BACKEND_URL` (env var or Streamlit secret) selects the backend; default `http://127.0.0.1:5000`.

## API Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | health check |
| POST | `/api/extract` | `{text, merge_duplicates?}` → items, keywords, statistics, summary |
| POST | `/api/analyze` | like extract + explainability; also accepts structured `items` |
| POST | `/api/compare` | `{list_a, list_b}` → common / only-in-A / only-in-B |
| POST | `/api/categorize` | `{items:[...]}` → category + method + confidence |
| POST | `/api/upload` | multipart CSV → normalised items + statistics |
| POST | `/api/report` | items → TXT and CSV report content |

## Example Input
`Please buy 2 kg rice, one litre milk, six apples, 3 packets of biscuits, 500 grams cheese and some tomatoes.`

## Example Output
| Item | Quantity | Unit | Category |
|---|---|---|---|
| Rice | 2 | kg | Grains |
| Milk | 1 | litre | Dairy |
| Apple | 6 | pieces | Fruits |
| Biscuit | 3 | packets | Snacks |
| Cheese | 500 | g | Dairy |
| Tomato | Not specified | — | Vegetables |

Total 6 · Specified 5 · Missing 1 · Categories 5 · Keywords: rice, milk, apple, biscuit, cheese, tomato

## Design notes (accuracy)
- "six apples" has a count but no unit → unit shown as *pieces* (marked `unit_inferred`). "a dozen/one dozen eggs" → 12 pieces.
- Eggs are grouped under **Dairy** (no separate egg category in the required list).
- Duplicates with the same/convertible unit are merged and a note says so (500 g + 1 kg = 1.5 kg); incompatible ones stay separate, flagged *Duplicate*.
- Negative, zero, malformed numbers and unsupported units are flagged, never silently replaced.
- Confidence is a heuristic score (dictionary evidence + quantity evidence), not a probability.
- PDF export is not included (kept dependency-free); TXT and CSV are provided.

## Testing
```bash
pip install pytest
python -m pytest -v
```
68 tests cover preprocessing, extraction, quantities, categories, duplicates, validation and the API.

## Screenshots
Add screenshots of each page to `screenshots/` and link them here.

## Deployment
- **Backend (Render/Railway/Heroku):** uses `Procfile` (`gunicorn backend.app:app`) and `runtime.txt`.
- **Frontend (Streamlit Community Cloud):** main file `frontend/app.py`; set secret `BACKEND_URL = "https://<your-backend>"`.
- Never hard-code a production URL; always use `BACKEND_URL`.

## Future Scope
POS-tagging / spaCy NER, multilingual input (Hindi/Marathi), brand and size parsing, learned category model, voice input, PDF export, price estimation.

## Conclusion
The project shows a complete NLP information-extraction pipeline — preprocessing, rule-based entity extraction, normalisation, classification and TF-IDF — delivered through a REST backend and an interactive dashboard, while staying strictly faithful to the user's input.
