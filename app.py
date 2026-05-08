
import streamlit as st
import math
import re
import time
from collections import Counter

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NLP Search Engine", page_icon="🔍", layout="wide")

STOP_WORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","and","but","or","nor","for","yet","so","in","on","at",
    "to","of","with","by","from","up","about","into","through","before",
    "after","above","below","between","each","all","both","few","more",
    "most","other","some","such","no","not","only","same","than","too",
    "very","just","as","it","its","this","that","these","those","we",
    "our","your","they","their","which","who","what","where","when","how",
    "using","like","use","help","make","many","also","allow","includes",
    "enables","involves","refers","focuses","provides","offers","manage",
    "find","build","handle","store","used","based","widely","popular",
}

BM25_K1 = 1.5
BM25_B  = 0.75

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_docs(path="data.txt"):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

@st.cache_data
def build_index(docs):
    """Tokenise docs and build BM25 index structures."""
    def tokenize(text):
        tokens = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
        return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]

    tokenized  = [tokenize(d) for d in docs]
    avg_dl     = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
    N          = len(docs)

    df: dict[str, int] = {}
    for tok in tokenized:
        for term in set(tok):
            df[term] = df.get(term, 0) + 1

    # Collect all unique words for autocomplete
    vocab = sorted({w for tok in tokenized for w in tok})
    return tokenized, avg_dl, N, df, vocab

@st.cache_data
def infer_categories(docs):
    """Rule-based category tags for each document."""
    rules = {
        "AI / ML":       ["machine learning","deep learning","neural","artificial intelligence",
                          "nlp","natural language","computer vision","supervised","unsupervised"],
        "Data":          ["data science","database","sql","nosql","mongodb","big data",
                          "hadoop","spark","warehouse","analytics","visualization","statistical"],
        "Web":           ["web","frontend","backend","react","javascript","html","css",
                          "search engine","information retrieval","ranking","interface","ux"],
        "Cloud":         ["cloud","aws","amazon","azure","microsoft","server","storage","internet"],
        "Security":      ["cybersecurity","encryption","firewall","network","attack","protect"],
        "Systems":       ["operating system","linux","windows","hardware","software resource"],
        "Mobile":        ["mobile","android","ios","flutter","kotlin"],
        "Languages":     ["java","python","c++","c language","kotlin","r language","programming language"],
        "Engineering":   ["software engineering","agile","git","version control","methodology"],
    }
    cats = []
    for doc in docs:
        low = doc.lower()
        matched = "General"
        for cat, kws in rules.items():
            if any(kw in low for kw in kws):
                matched = cat
                break
        cats.append(matched)
    return cats

# ─── BM25 SCORING ─────────────────────────────────────────────────────────────
def bm25_search(query: str, tokenized, avg_dl, N, df, docs, cats,
                top_k=8, filter_cat=None):
    def tokenize(text):
        tokens = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
        return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]

    q_terms = tokenize(query)
    if not q_terms:
        return []

    scores = []
    for idx, tok in enumerate(tokenized):
        if filter_cat and cats[idx] != filter_cat:
            continue
        dl   = len(tok)
        tf_c = Counter(tok)
        score = 0.0
        for term in q_terms:
            if term not in df:
                continue
            tf  = tf_c.get(term, 0)
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            num = tf * (BM25_K1 + 1)
            den = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl)
            score += idf * (num / den)
        if score > 0:
            scores.append((idx, score))

    scores.sort(key=lambda x: -x[1])
    max_s = scores[0][1] if scores else 1
    return [(docs[i], cats[i], round(s / max_s * 100)) for i, s in scores[:top_k]]

# ─── HIGHLIGHTING ─────────────────────────────────────────────────────────────
def highlight(text: str, query: str) -> str:
    terms = re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
    terms = [t for t in terms if t not in STOP_WORDS and len(t) > 1]
    for term in terms:
        text = re.sub(
            f"({re.escape(term)})",
            r"**\1**",
            text,
            flags=re.IGNORECASE,
        )
    return text

# ─── AUTOCOMPLETE ─────────────────────────────────────────────────────────────
def get_suggestions(prefix: str, vocab: list[str], n=8) -> list[str]:
    p = prefix.lower()
    starts = [w for w in vocab if w.startswith(p) and w != p]
    contains = [w for w in vocab if p in w and not w.startswith(p)]
    return (starts + contains)[:n]

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .result-card {
      background: #f9f9f9;
      border: 1px solid #e0e0e0;
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 10px;
  }
  .score-badge {
      display: inline-block;
      background: #e8f4fd;
      color: #1a73e8;
      border-radius: 6px;
      padding: 2px 10px;
      font-size: 13px;
      font-weight: 600;
  }
  .cat-badge {
      display: inline-block;
      background: #f0f4ff;
      color: #5c6bc0;
      border-radius: 6px;
      padding: 2px 8px;
      font-size: 12px;
      margin-left: 8px;
  }
  .rank-num {
      color: #888;
      font-size: 13px;
      margin-right: 6px;
  }
  .progress-bar {
      height: 6px;
      border-radius: 4px;
      background: #e0e0e0;
      margin-top: 6px;
  }
  .progress-fill {
      height: 6px;
      border-radius: 4px;
      background: linear-gradient(90deg, #1a73e8, #34a853);
  }
</style>
""", unsafe_allow_html=True)

# ─── LOAD & BUILD ─────────────────────────────────────────────────────────────
docs     = load_docs()
cats     = infer_categories(docs)
tokenized, avg_dl, N, df, vocab = build_index(docs)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.slider("Max results", 3, 15, 8)
    cat_options = ["All"] + sorted(set(cats))
    filter_cat  = st.selectbox("Filter by category", cat_options)
    filter_cat  = None if filter_cat == "All" else filter_cat
    st.markdown("---")
    st.caption("**Algorithm:** BM25 (Okapi)\n\n**Index size:** {} docs\n\n**Vocab:** {} terms".format(
        len(docs), len(vocab)))

# ─── MAIN UI ──────────────────────────────────────────────────────────────────
st.title("🔍 NLP Search Engine")

# Search input
query = st.text_input("", placeholder="Search for topics… e.g. machine learning, cloud security",
                       label_visibility="collapsed")

# Autocomplete suggestions
if query and len(query) >= 2:
    suggs = get_suggestions(query.split()[-1], vocab)
    if suggs:
        st.caption("💡 Suggestions: " + "  ·  ".join(f"`{s}`" for s in suggs))

# Search & display
if query:
    t0      = time.perf_counter()
    results = bm25_search(query, tokenized, avg_dl, N, df, docs, cats,
                          top_k=top_k, filter_cat=filter_cat)
    elapsed = (time.perf_counter() - t0) * 1000

    if results:
        st.markdown(f"**{len(results)} result{'s' if len(results)!=1 else ''}** &nbsp;·&nbsp; "
                    f"<small style='color:#888'>{elapsed:.1f} ms</small>", unsafe_allow_html=True)
        st.markdown("")

        for rank, (doc, cat, pct) in enumerate(results, 1):
            highlighted = highlight(doc, query)
            bar_html = (
                f'<div class="progress-bar">'
                f'<div class="progress-fill" style="width:{pct}%"></div>'
                f'</div>'
            )
            st.markdown(
                f'<div class="result-card">'
                f'<span class="rank-num">#{rank}</span>'
                f'<span class="score-badge">{pct}%</span>'
                f'<span class="cat-badge">{cat}</span>'
                f'<p style="margin:8px 0 4px;font-size:15px;">{highlighted}</p>'
                f'{bar_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("😕 No results found. Try different keywords.")
else:
    st.info("Enter a search term above to get started.")