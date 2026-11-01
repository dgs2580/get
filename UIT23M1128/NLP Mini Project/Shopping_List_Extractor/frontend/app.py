"""Streamlit frontend for SHOPPING LIST EXTRACTOR.   Run: streamlit run frontend/app.py

The UI contains no NLP logic: every result comes from the Flask REST backend.
"""
import html
import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "").strip()
if not BACKEND_URL:
    try:
        BACKEND_URL = str(st.secrets.get("BACKEND_URL", "")).strip()
    except Exception:
        BACKEND_URL = ""
BACKEND_URL = (BACKEND_URL or "http://127.0.0.1:5000").rstrip("/")
TIMEOUT = 20

PRIMARY, SECONDARY, TEXT = "#2563EB", "#16A34A", "#1E293B"
WARNING, DANGER, BORDER, CARD, BG = "#F59E0B", "#DC2626", "#E2E8F0", "#FFFFFF", "#F8FAFC"
SUBTITLE = "Convert natural-language shopping requests into a structured shopping list using NLP."
EXAMPLE = "Example: Buy 2 kg rice, 1 litre milk, 6 apples and 2 packets of biscuits."

st.set_page_config(page_title="Shopping List Extractor", page_icon="🛒", layout="wide")

st.markdown(f"""
<style>
.stApp {{ background:{BG}; color:{TEXT}; }}
[data-testid="stSidebar"] {{ background:{CARD}; border-right:1px solid {BORDER}; }}
h1,h2,h3,h4 {{ color:{TEXT}; }}
.card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:12px; padding:16px 18px; margin-bottom:12px; }}
.card h4 {{ margin:0 0 6px 0; color:{PRIMARY}; font-size:1.05rem; }}
.card p {{ margin:0; color:#475569; font-size:.92rem; }}
.item-card {{ background:{CARD}; border:1px solid {BORDER}; border-left:4px solid {PRIMARY};
             border-radius:10px; padding:12px 14px; margin-bottom:10px; }}
.item-card .name {{ font-weight:700; font-size:1.05rem; }}
.item-card .qty {{ color:{PRIMARY}; font-weight:600; }}
.item-card .meta {{ color:#64748B; font-size:.85rem; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; }}
.pill.pending {{ background:#FEF3C7; color:#92400E; }}
.pill.done {{ background:#DCFCE7; color:#166534; }}
.flow {{ background:{CARD}; border:1px dashed {BORDER}; border-radius:10px; padding:10px 14px;
        font-family:monospace; font-size:.85rem; line-height:1.5; color:{TEXT}; }}
.cat-title {{ color:{SECONDARY}; font-weight:700; letter-spacing:.04em; margin-top:10px; }}
div[data-testid="stMetric"] {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:10px 14px; }}
.stButton>button[kind="primary"] {{ background:{PRIMARY}; border-color:{PRIMARY}; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API client
def call_api(method, path, **kwargs):
    """Return (json, error_message). Never raises, never shows stack traces."""
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError:
        return None, (f"Backend unavailable at {BACKEND_URL}. Start it with "
                      "`python backend/app.py` and try again.")
    except requests.exceptions.Timeout:
        return None, "The backend took too long to respond (timeout). Please try again."
    except requests.exceptions.RequestException:
        return None, "A network problem occurred while contacting the backend."
    try:
        data = resp.json()
    except ValueError:
        return None, "The backend returned an unexpected response."
    if not isinstance(data, dict) or data.get("status") not in ("success", "error"):
        return None, "The backend returned an unexpected response."
    if data["status"] == "error":
        return None, data.get("message", "The request could not be processed.")
    return data, None


def esc(x):
    return html.escape(str(x))


# ------------------------------------------------------------------ state
S = st.session_state
S.setdefault("result", None)
S.setdefault("source_text", "")
S.setdefault("run_id", 0)
S.setdefault("manual_items", [])
S.setdefault("manual_result", None)
S.setdefault("csv_result", None)
S.setdefault("text_input", "")


def purchased(i):
    return bool(S.get(f"chk_{S.run_id}_{i}", False))


# ------------------------------------------------------------------ components
def show_messages(result):
    for w in result.get("warnings", []):
        st.warning(w, icon="⚠️")
    for n in result.get("merge_notes", []):
        st.info(n, icon="🔗")


def metric_row(stats):
    c = st.columns(5)
    c[0].metric("Total Items", stats["total_items"])
    c[1].metric("Specified Quantities", stats["specified_quantities"])
    c[2].metric("Missing Quantities", stats["missing_quantities"])
    c[3].metric("Categories", stats["categories"])
    c[4].metric("Duplicate Items", stats["duplicate_items"])


def items_dataframe(items):
    return pd.DataFrame([{
        "Item": i["display_name"],
        "Quantity": "Not specified" if i["quantity"] is None else f"{i['quantity']:g}",
        "Unit": i["unit_display"] if i["unit"] and i["quantity"] is not None else "—",
        "Category": i["category"], "Status": i["status"],
        "Confidence": i["confidence"], "Original phrase": i["original_phrase"],
    } for i in items])


def render_table(items):
    st.dataframe(items_dataframe(items), width="stretch", hide_index=True,
                 column_config={"Confidence": st.column_config.ProgressColumn(
                     "Confidence (heuristic)", min_value=0, max_value=1, format="%.2f")})


def render_cards(items):
    cols = st.columns(3)
    for idx, i in enumerate(items):
        done = purchased(idx)
        pill = '<span class="pill done">Purchased</span>' if done else '<span class="pill pending">Pending</span>'
        cols[idx % 3].markdown(
            f'<div class="item-card"><div class="name">{esc(i["display_name"])}</div>'
            f'<div class="qty">{esc(i["quantity_display"])}</div>'
            f'<div class="meta">Category: {esc(i["category"])}</div>{pill}</div>',
            unsafe_allow_html=True)


def render_checklist(items):
    st.markdown("#### Shopping checklist")
    for idx, i in enumerate(items):
        st.checkbox(f"{i['display_name']} — {i['quantity_display']}", key=f"chk_{S.run_id}_{idx}")
    done = sum(purchased(i) for i in range(len(items)))
    st.caption(f"Purchased: {done}  |  Pending: {len(items) - done}")


def render_category_sections(category_summary):
    for cat, rows in category_summary.items():
        st.markdown(f'<div class="cat-title">{esc(cat.upper())}</div>', unsafe_allow_html=True)
        for r in rows:
            st.markdown(f"- {r['item']} — {r['quantity_display']}")


def render_charts(stats, key):
    counts = stats.get("category_counts", {})
    if not counts:
        st.info("No data available for charts.")
        return
    df = pd.DataFrame({"Category": list(counts), "Items": list(counts.values())})
    c1, c2 = st.columns(2)
    fig = px.bar(df, x="Category", y="Items", title="Items by Category",
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(plot_bgcolor=CARD, paper_bgcolor=CARD, font_color=TEXT, yaxis=dict(dtick=1))
    c1.plotly_chart(fig, width="stretch", key=f"bar_{key}")
    fig = px.pie(df, names="Category", values="Items", hole=0.5, title="Category Distribution",
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(paper_bgcolor=CARD, font_color=TEXT)
    c2.plotly_chart(fig, width="stretch", key=f"pie_{key}")
    q = pd.DataFrame({"Quantity": ["Specified", "Missing", "Invalid"],
                      "Items": [stats["specified_quantities"],
                                stats["missing_quantities"] - stats["invalid_quantities"],
                                stats["invalid_quantities"]]})
    fig = px.bar(q, x="Quantity", y="Items", title="Quantity Availability", color="Quantity",
                 color_discrete_map={"Specified": SECONDARY, "Missing": WARNING, "Invalid": DANGER})
    fig.update_layout(plot_bgcolor=CARD, paper_bgcolor=CARD, font_color=TEXT,
                      showlegend=False, yaxis=dict(dtick=1))
    st.plotly_chart(fig, width="stretch", key=f"qty_{key}")


def render_keywords(keywords):
    st.markdown("#### Top Keywords — Relative Keyword Importance")
    st.caption("TF-IDF scores show relative importance of terms in this request. They are not probabilities.")
    if not keywords:
        st.info("No keywords available.")
        return
    df = pd.DataFrame(keywords).rename(columns={
        "keyword": "Keyword", "tf": "TF", "idf": "IDF", "tfidf": "TF × IDF",
        "relative_importance": "Relative importance"})
    st.dataframe(df, width="stretch", hide_index=True,
                 column_config={"Relative importance": st.column_config.ProgressColumn(
                     min_value=0, max_value=1, format="%.2f")})


def render_downloads(result, source_text=None, prefix="dl"):
    data, err = call_api("POST", "/api/report", json={
        "items": result["items"], "statistics": result["statistics"],
        "keywords": result["keywords"], "original_text": source_text})
    if err:
        st.error(err)
        return
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("⬇ Download Shopping List (TXT)", data["txt"], "shopping_list.txt",
                       "text/plain", key=f"{prefix}_txt")
    c2.download_button("⬇ Download Shopping List (CSV)", data["csv"], "shopping_list.csv",
                       "text/csv", key=f"{prefix}_csv")


def render_full_result(result, source_text=None, prefix="r"):
    show_messages(result)
    metric_row(result["statistics"])
    st.markdown("### Extracted Shopping List")
    render_table(result["items"])
    st.info(result["summary"])
    st.markdown("### Item cards")
    render_cards(result["items"])
    left, right = st.columns(2)
    with left:
        render_checklist(result["items"])
    with right:
        st.markdown("#### Category-wise summary")
        render_category_sections(result["category_summary"])
    render_keywords(result["keywords"])
    render_downloads(result, source_text, prefix)


# ------------------------------------------------------------------ pages
def page_dashboard():
    st.title("SHOPPING LIST EXTRACTOR")
    st.write(SUBTITLE)
    cards = [("NLP-Based Extraction", "NLTK tokenisation, lemmatisation and rule-based entity extraction."),
             ("Quantity Detection", "Numbers, number words, fractions, dozen/half and units like kg or litre."),
             ("Smart Categorization", "Dictionary + head-noun rules + guarded TF-IDF similarity."),
             ("Structured Shopping List", "Clean table, category groups, checklist and downloads.")]
    cols = st.columns(4)
    for col, (t, d) in zip(cols, cards):
        col.markdown(f'<div class="card"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    st.markdown("### How it works")
    st.markdown('<div class="flow">Input → Process → Extract → Organize → Export</div>', unsafe_allow_html=True)
    st.write("")
    data, err = call_api("GET", "/api/health")
    if err:
        st.error(err)
    else:
        st.success(f"Backend connected ({BACKEND_URL})")
    st.caption("Go to **Extract Shopping List** in the sidebar to try it.")


def page_extract():
    st.header("Extract Shopping List")
    st.write("Enter your shopping request in natural language. The NLP engine will identify "
             "items, quantities, units and categories.")
    st.text_area("Shopping request", key="text_input", height=140, placeholder=EXAMPLE,
                 label_visibility="collapsed")
    merge = st.checkbox("Combine duplicate items when units are compatible", value=True)
    c1, c2, _ = st.columns([1.3, 0.6, 4])
    go = c1.button("Extract Shopping List", type="primary")
    clear = c2.button("Clear")
    if clear:
        S.result, S.source_text = None, ""
        S.pop("text_input", None)
        S.run_id += 1
        st.rerun()
    if go:
        text = S.text_input
        if not text.strip():
            st.error("Input is empty. Please type your shopping request.")
        else:
            with st.spinner("Running NLP pipeline..."):
                data, err = call_api("POST", "/api/analyze", json={"text": text, "merge_duplicates": merge})
            if err:
                st.error(err)
            else:
                S.result, S.source_text = data, text
                S.run_id += 1
    if S.result:
        render_full_result(S.result, S.source_text, "extract")


def page_manual():
    st.header("Manual List")
    st.write("Structured input for comparison with NLP extraction. The NLP page is the main feature.")
    units = [""] + ["kg", "g", "mg", "litre", "ml", "packet", "bottle", "box", "piece", "unit", "can", "jar", "tube"]
    cats = ["(auto-detect)", "Fruits", "Vegetables", "Dairy", "Grains", "Bakery", "Meat", "Seafood", "Beverages",
            "Snacks", "Spices", "Condiments", "Frozen Foods", "Personal Care", "Household", "Cleaning",
            "Stationery", "Other"]
    c = st.columns([2, 1, 1, 1.5])
    item = c[0].text_input("Item", key="m_item")
    qty = c[1].text_input("Quantity", key="m_qty", placeholder="optional")
    unit = c[2].selectbox("Unit", units, key="m_unit")
    cat = c[3].selectbox("Category", cats, key="m_cat")
    b1, b2, _ = st.columns([1, 1, 5])
    if b1.button("Add Item", type="primary"):
        if not item.strip():
            st.error("Item name is required.")
        else:
            S.manual_items.append({"item": item, "quantity": qty or None, "unit": unit or None,
                                   "category": None if cat == "(auto-detect)" else cat})
    if b2.button("Clear list"):
        S.manual_items, S.manual_result = [], None
        st.rerun()
    if S.manual_items:
        data, err = call_api("POST", "/api/analyze", json={"items": S.manual_items, "merge_duplicates": True})
        if err:
            st.error(err)
        else:
            S.manual_result = data
            render_full_result(data, None, "manual")
    else:
        st.info("No items added yet.")


def page_csv():
    st.header("CSV Analysis")
    st.write("Upload Shopping List CSV. Accepted columns: item / item_name / product, quantity / qty, "
             "unit / units, category / category_name.")
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "sample_shopping_lists.csv"), "rb") as f:
            st.download_button("Download sample CSV", f.read(), "sample_shopping_lists.csv", "text/csv")
    except OSError:
        pass
    up = st.file_uploader("Upload Shopping List CSV", type=["csv"])
    if up is not None:
        with st.spinner("Analysing CSV..."):
            data, err = call_api("POST", "/api/upload", files={"file": (up.name, up.getvalue(), "text/csv")})
        if err:
            st.error(err)
            S.csv_result = None
        else:
            S.csv_result = data
    r = S.csv_result
    if r:
        show_messages(r)
        s = r["statistics"]
        c = st.columns(6)
        c[0].metric("Rows", r["rows"])
        c[1].metric("Columns", len(r["columns"]))
        c[2].metric("Total Items", s["total_items"])
        c[3].metric("Categories", s["categories"])
        c[4].metric("Missing Quantities", s["missing_quantities"])
        c[5].metric("Duplicate Items", s["duplicate_items"])
        st.markdown("#### Dataset preview")
        st.dataframe(pd.DataFrame(r["preview"]), width="stretch", hide_index=True)
        st.markdown("#### Normalised items")
        render_table(r["items"])
        render_charts(s, "csv")
        render_downloads(r, None, "csv")


def page_compare():
    st.header("Compare Lists")
    st.write("Compare two shopping lists written in natural language.")
    a, b = st.columns(2)
    la = a.text_area("LIST A", height=130, placeholder="Rice, milk and apples", key="cmp_a")
    lb = b.text_area("LIST B", height=130, placeholder="Rice, milk, bread and eggs", key="cmp_b")
    if st.button("Compare Lists", type="primary"):
        if not la.strip() or not lb.strip():
            st.error("Please fill in both lists.")
            return
        data, err = call_api("POST", "/api/compare", json={"list_a": la, "list_b": lb})
        if err:
            st.error(err)
            return
        c = data["comparison"]
        m = st.columns(4)
        m[0].metric("Items in A", c["counts"]["list_a"])
        m[1].metric("Items in B", c["counts"]["list_b"])
        m[2].metric("Common", c["counts"]["common"])
        m[3].metric("Different", c["counts"]["only_a"] + c["counts"]["only_b"])
        x, y, z = st.columns(3)
        x.markdown("#### Common Items")
        for r in c["common"]:
            note = "" if r["same_quantity"] else f" (A: {r['list_a']} | B: {r['list_b']})"
            x.markdown(f"- {r['item']}{note}")
        y.markdown("#### Only in List A")
        for r in c["only_in_a"]:
            y.markdown(f"- {r['item']}")
        z.markdown("#### Only in List B")
        for r in c["only_in_b"]:
            z.markdown(f"- {r['item']}")


def current_result():
    return S.result or S.manual_result or S.csv_result


def page_stats():
    st.header("Statistics")
    r = current_result()
    if not r:
        st.info("Nothing to show yet. Extract a shopping list, add manual items or upload a CSV first.")
        return
    s = r["statistics"]
    metric_row(s)
    c = st.columns(2)
    c[0].metric("Most Common Category", s["most_common_category"])
    c[1].metric("Invalid Quantities", s["invalid_quantities"])
    render_charts(s, "stats")


def page_explain():
    st.header("NLP Explainability")
    r = S.result
    if not r or "explainability" not in r:
        st.info("Run an extraction on the **Extract Shopping List** page to see each NLP step here.")
        return
    ex = r["explainability"]
    st.markdown("**1. Original text**")
    st.code(ex["original_text"], language=None)
    st.markdown("**2. Cleaned text**")
    st.code(ex["cleaned_text"], language=None)
    st.markdown("**3. Sentences (NLTK sentence segmentation)**")
    st.write(ex["sentences"])
    st.markdown("**4. Tokens (NLTK word tokenization)**")
    st.write(" · ".join(ex["tokens"]))
    st.markdown("**5. Tokens after stopword removal** (numbers, units and quantity words preserved)")
    st.write(" · ".join(ex["tokens_without_stopwords"]))
    st.markdown("**6. Lemmas (WordNet lemmatizer)**")
    st.write(" · ".join(ex["lemmas"]))
    st.markdown("**7. Item segments** found by rules")
    st.write(ex["segments"])
    if ex["ignored_phrases"]:
        st.caption("Ignored group phrases (not items): " + ", ".join(ex["ignored_phrases"]))
    st.markdown("**8. Extraction trace** — items, quantities, units, categories")
    st.dataframe(pd.DataFrame(ex["extraction_trace"]), width="stretch", hide_index=True)
    st.markdown("**9. Quantity associations**")
    for i in r["items"]:
        st.write(f"{i['item']} → {i['quantity_display']}")
    render_keywords(r["keywords"])


def page_about():
    st.header("About Project")
    st.write("**Shopping List Extractor** is a college NLP project that turns an unstructured shopping "
             "request into a structured list using Information Extraction.")
    steps = [("Input", "User provides a natural-language shopping request."),
             ("Preprocessing", "Text is cleaned, split into sentences and tokenized (NLTK)."),
             ("Quantity Extraction", "Numbers, number words, fractions and dozen/half/quarter are identified."),
             ("Unit Extraction", "Units such as kg, litre and packet are identified and standardised."),
             ("Item Extraction", "Shopping/product entities are found with rules over the tokens."),
             ("Normalization", "Items are lemmatized (apples → apple); units are made consistent."),
             ("Category Classification", "Dictionary lookup, head-noun rule, guarded TF-IDF similarity; else Other."),
             ("TF-IDF", "scikit-learn TF-IDF ranks important terms (relative importance)."),
             ("Structured Output", "Everything is returned as JSON and shown as a shopping list.")]
    for n, (t, d) in enumerate(steps, 1):
        st.markdown(f"**Step {n} — {t}:** {d}")
    st.markdown("**Technology:** Python, Streamlit, Flask, NLTK, scikit-learn, pandas, NumPy, Plotly.")
    st.markdown("**No-fabrication rule:** the app never invents items, quantities, units or categories. "
                "Missing values are shown as *Not specified*; unknown items as *Other*.")


# ------------------------------------------------------------------ sidebar / routing
PAGES = {"Dashboard": page_dashboard, "Extract Shopping List": page_extract, "Manual List": page_manual,
         "CSV Analysis": page_csv, "Compare Lists": page_compare, "Statistics": page_stats,
         "NLP Explainability": page_explain, "About Project": page_about}

with st.sidebar:
    st.markdown("## SHOPPING LIST EXTRACTOR")
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**NLP PIPELINE**")
    st.markdown('<div class="flow">Text Input<br>↓<br>Preprocessing<br>↓<br>Tokenization<br>↓<br>'
                'Item Extraction<br>↓<br>Quantity Extraction<br>↓<br>Normalization<br>↓<br>'
                'Category Classification<br>↓<br>Structured List</div>', unsafe_allow_html=True)

PAGES[page]()
