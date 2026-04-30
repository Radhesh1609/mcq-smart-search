"""
MCQ Smart Search  ·  Mobile-First  ·  Complete Edition
========================================================
• Upload MCQ files (PDF / TXT)
• Type any keyword → instant suggestions appear below
• Tap a suggestion → full answer with highlighted correct option
• Phone-frame layout, dark purple theme

Install:
    pip install streamlit pdfplumber scikit-learn numpy

Run:
    streamlit run app.py
"""

import re, io, textwrap
import numpy as np
import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ════════════════════════════════════════════════════════
# PAGE CONFIG  ← must be very first Streamlit call
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MCQ Search",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════
# GLOBAL CSS  — dark purple phone layout
# ════════════════════════════════════════════════════════
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">

<style>
/* ──────────── RESET ──────────── */
*, *::before, *::after { box-sizing: border-box; margin:0; padding:0; }

html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
    background: #08050f !important;
    color: #e8deff;
}

/* hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

/* phone-width centering */
.block-container {
    max-width: 440px !important;
    padding: 0 0 90px 0 !important;
    margin: 0 auto !important;
}

/* ──────────── TOP NAV TABS ──────────── */
/* we style the two tab-switch buttons at the very top */
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
    background: #110a22 !important;
    border: 1.5px solid rgba(160,100,255,0.18) !important;
    border-radius: 10px !important;
    color: #7a5aaa !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    padding: 9px 6px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    margin-bottom: 0 !important;
    transform: none !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
    background: #1a0d35 !important;
    border-color: #a064ff !important;
    color: #c99dff !important;
    transform: none !important;
}

/* ──────────── HEADER BLOCK ──────────── */
.app-header {
    background: linear-gradient(175deg, #120428 0%, #1c0850 60%, #0e0520 100%);
    padding: 18px 20px 16px;
    border-bottom: 1px solid rgba(160,100,255,0.15);
}
.hdr-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
}
.app-logo {
    font-size: 20px;
    font-weight: 900;
    color: #fff;
    letter-spacing: -0.5px;
}
.app-logo span { color: #b57bff; }
.q-badge {
    background: rgba(181,123,255,0.15);
    border: 1px solid rgba(181,123,255,0.3);
    border-radius: 20px;
    padding: 4px 13px;
    font-size: 11px;
    font-weight: 700;
    color: #c99dff;
}

/* ──────────── SEARCH INPUT ──────────── */
[data-testid="stTextInput"] { margin: 0 !important; }
[data-testid="stTextInput"] > label { display: none !important; }
[data-testid="stTextInput"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 2px solid rgba(181,123,255,0.3) !important;
    border-radius: 16px !important;
    padding: 0 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] > div:focus-within {
    border-color: #b57bff !important;
    box-shadow: 0 0 0 4px rgba(181,123,255,0.12) !important;
    background: rgba(255,255,255,0.09) !important;
}
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #f0eaff !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 14px 18px !important;
    caret-color: #b57bff !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #4a3566 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* ──────────── SUGGESTION BUTTONS ──────────── */
.sugg-wrap { padding: 12px 16px 4px; }

.sugg-label {
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #4a3566;
    margin-bottom: 10px;
}

/* every stButton inside sugg-wrap */
.sugg-wrap .stButton > button {
    background: #120928 !important;
    border: 1.5px solid rgba(181,123,255,0.18) !important;
    border-radius: 16px !important;
    color: #d4c0ff !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    padding: 14px 16px !important;
    width: 100% !important;
    text-align: left !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    line-height: 1.5 !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 52px !important;
    margin-bottom: 0 !important;
}
.sugg-wrap .stButton > button:hover {
    background: #1c0e40 !important;
    border-color: #b57bff !important;
    color: #ecdeff !important;
    transform: translateX(4px) !important;
    box-shadow: 0 4px 22px rgba(181,123,255,0.2) !important;
}
.sugg-wrap .stButton > button:active {
    transform: scale(0.97) translateX(2px) !important;
}

.meta-row {
    display: flex;
    gap: 6px;
    align-items: center;
    padding: 5px 4px 10px;
}
.badge {
    border-radius: 20px;
    padding: 2px 9px;
    font-size: 10px;
    font-weight: 800;
}
.badge-match  { background: rgba(181,123,255,0.15); color: #c099ff; }
.badge-ok     { background: rgba(74,222,128,0.12);  color: #4ade80; }
.badge-none   { background: rgba(251,146,60,0.12);  color: #fb923c; }
.badge-opts   { background: rgba(56,189,248,0.1);   color: #38bdf8; }

.sugg-divider { height: 1px; background: rgba(255,255,255,0.04); margin: 0 2px 2px; }

/* ──────────── ANSWER SCREEN ──────────── */
.ans-wrap { padding: 14px 16px; }

/* back button — reuse sugg-wrap style but override */
.back-wrap .stButton > button {
    background: rgba(181,123,255,0.1) !important;
    border: 1.5px solid rgba(181,123,255,0.25) !important;
    border-radius: 20px !important;
    color: #c99dff !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    padding: 7px 18px !important;
    width: auto !important;
    cursor: pointer !important;
    margin-bottom: 14px !important;
    transform: none !important;
}
.back-wrap .stButton > button:hover {
    background: rgba(181,123,255,0.2) !important;
    transform: none !important;
    box-shadow: none !important;
}

.q-card {
    background: linear-gradient(135deg, #130630 0%, #1e0a48 100%);
    border: 1.5px solid rgba(181,123,255,0.28);
    border-radius: 20px;
    padding: 18px 18px 20px;
    margin-bottom: 14px;
}
.q-card-eyebrow {
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6a4a99;
    margin-bottom: 9px;
}
.q-card-text {
    font-size: 15.5px;
    font-weight: 700;
    color: #f0e8ff;
    line-height: 1.6;
}

.opts-section { display: flex; flex-direction: column; gap: 9px; margin-bottom: 14px; }

.opt-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 13px 14px;
    border-radius: 14px;
    border: 1.5px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.03);
    transition: all 0.15s;
}
.opt-correct {
    background: rgba(74,222,128,0.09) !important;
    border-color: rgba(74,222,128,0.45) !important;
}
.opt-letter {
    width: 30px;
    height: 30px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 800;
    flex-shrink: 0;
    background: rgba(255,255,255,0.05);
    color: #7a5aaa;
    border: 1px solid rgba(255,255,255,0.09);
    font-family: monospace;
}
.opt-letter-c {
    background: #22c55e !important;
    color: #fff !important;
    border-color: transparent !important;
}
.opt-text {
    font-size: 13.5px;
    color: #8a7aaa;
    line-height: 1.5;
    font-weight: 500;
    flex: 1;
    padding-top: 5px;
}
.opt-text-c {
    color: #86efac !important;
    font-weight: 700 !important;
}
.opt-check {
    margin-left: auto;
    color: #4ade80;
    font-size: 17px;
    padding-top: 4px;
    flex-shrink: 0;
}

.expl-box {
    background: rgba(251,191,36,0.07);
    border-left: 3.5px solid #f59e0b;
    border-radius: 0 14px 14px 0;
    padding: 13px 16px;
    margin-bottom: 14px;
}
.expl-eyebrow {
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #ca8a04;
    margin-bottom: 6px;
}
.expl-text {
    font-size: 13px;
    color: #fde68a;
    font-weight: 500;
    line-height: 1.65;
}

.ans-only-box {
    display: flex;
    align-items: center;
    gap: 14px;
    background: rgba(74,222,128,0.08);
    border: 1.5px solid rgba(74,222,128,0.35);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 14px;
}
.ans-only-label { font-size: 10px; font-weight: 800; color: #4ade80; letter-spacing: 1px; text-transform: uppercase; }
.ans-only-val   { font-size: 28px; font-weight: 900; color: #86efac; line-height: 1; }

.no-opt-msg { font-size: 12px; color: #4a3566; padding: 10px 0; font-weight: 600; }

/* ──────────── UPLOAD SECTION ──────────── */
.upload-wrap { padding: 16px 16px 0; }
.upload-hdr {
    background: linear-gradient(175deg, #120428 0%, #1c0850 100%);
    padding: 18px 20px 18px;
    border-bottom: 1px solid rgba(160,100,255,0.15);
    margin-bottom: 0;
}
.upload-hdr-title { font-size: 20px; font-weight: 900; color: #fff; }
.upload-hdr-title span { color: #b57bff; }

.stat-strip { display: flex; gap: 8px; margin-bottom: 16px; margin-top: 16px; }
.stat-box {
    flex: 1;
    background: #120928;
    border: 1.5px solid rgba(181,123,255,0.18);
    border-radius: 14px;
    padding: 14px 10px;
    text-align: center;
}
.stat-num { font-size: 24px; font-weight: 900; color: #b57bff; line-height: 1; margin-bottom: 5px; }
.stat-lbl { font-size: 9px; font-weight: 800; color: #4a3566; text-transform: uppercase; letter-spacing: 1px; }

.upload-sec-label {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #7a5aaa;
    margin: 14px 0 7px;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* file uploader override */
[data-testid="stFileUploader"] {
    background: #120928 !important;
    border: 1.5px dashed rgba(181,123,255,0.25) !important;
    border-radius: 14px !important;
    padding: 10px 14px !important;
    margin-bottom: 12px !important;
}
[data-testid="stFileUploader"] * {
    color: #7a5aaa !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; }

/* process button */
.proc-btn .stButton > button {
    background: linear-gradient(135deg, #6d28d9 0%, #a855f7 100%) !important;
    border: none !important;
    border-radius: 16px !important;
    color: #fff !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    padding: 16px !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 8px 28px rgba(109,40,217,0.45) !important;
    transition: opacity 0.2s, box-shadow 0.2s !important;
    transform: none !important;
    margin-top: 4px !important;
}
.proc-btn .stButton > button:hover {
    opacity: 0.92 !important;
    box-shadow: 0 10px 35px rgba(109,40,217,0.6) !important;
    transform: none !important;
}

/* ──────────── EMPTY / NO-RESULT STATES ──────────── */
.empty-wrap {
    text-align: center;
    padding: 48px 24px 32px;
}
.empty-orb {
    width: 76px; height: 76px;
    background: rgba(181,123,255,0.1);
    border: 2px solid rgba(181,123,255,0.2);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 32px;
    margin: 0 auto 18px;
}
.empty-title { font-size: 16px; font-weight: 800; color: #c9b0ff; margin-bottom: 9px; }
.empty-sub   { font-size: 12.5px; color: #4a3566; font-weight: 500; line-height: 1.7; }

.no-res-wrap { text-align: center; padding: 36px 20px; }
.no-res-icon { font-size: 36px; margin-bottom: 12px; }
.no-res-title { font-size: 15px; font-weight: 700; color: #7a5aaa; margin-bottom: 7px; }
.no-res-sub   { font-size: 12px; color: #4a3566; font-weight: 500; }

/* ──────────── BOTTOM TAB BAR ──────────── */
.bottom-bar {
    position: fixed;
    bottom: 0;
    left: 50%; transform: translateX(-50%);
    width: 100%; max-width: 440px;
    background: #0c0618;
    border-top: 1px solid rgba(181,123,255,0.12);
    display: flex;
    z-index: 1000;
    padding: 10px 0 18px;
}
.tab-btn {
    flex: 1;
    display: flex; flex-direction: column;
    align-items: center; gap: 3px;
}
.tab-icon { font-size: 22px; }
.tab-lbl  { font-size: 10px; font-weight: 800; letter-spacing: 0.5px; }
.tab-on  .tab-lbl { color: #b57bff; }
.tab-off .tab-lbl { color: #3a2555; }

/* alerts */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ════════════════════════════════════════════════════════

def extract_pdf(data: bytes) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    pages = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
    except Exception:
        pass
    return "\n".join(pages)


def extract_txt(data: bytes) -> str:
    """Decode a text file, trying common encodings."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return ""


# ════════════════════════════════════════════════════════
# MCQ PARSER
# ════════════════════════════════════════════════════════

def parse_mcq(raw: str) -> list:
    """
    Detect MCQ blocks from raw text.
    Handles formats:
      1. Question text
         A) option   B) option   Answer: A
      Q1. / Question 1: / just blank-line paragraphs (fallback)
    """
    questions = []

    # Primary: numbered blocks
    block_re = re.compile(
        r'(?:^|\n)\s*(?:Q(?:uestion)?\s*)?(\d+)\s*[.):\-]\s*(.+?)'
        r'(?=\n\s*(?:Q(?:uestion)?\s*)?\d+\s*[.):\-]|\Z)',
        re.DOTALL | re.IGNORECASE,
    )
    blocks = block_re.findall(raw)

    # Fallback: blank-line paragraph split
    if len(blocks) < 2:
        paras  = [p.strip() for p in re.split(r'\n{2,}', raw.strip()) if len(p.strip()) > 25]
        blocks = [(str(i+1), p) for i, p in enumerate(paras)]

    opt_re = re.compile(r'^\s*[\(\[]?([A-Ea-e])[\).\]\-\s]+(.+)', re.MULTILINE)
    ans_re = re.compile(
        r'(?:answer|ans|correct(?:\s+ans(?:wer)?)?|key)\s*[:\-–]\s*[\(]?([A-Ea-e])[\)]?',
        re.IGNORECASE,
    )
    exp_re = re.compile(
        r'(?:explanation|reason|note|hint|solution)\s*[:\-–]\s*(.+)',
        re.IGNORECASE | re.DOTALL,
    )

    for _, body in blocks:
        body = body.strip()
        if not body or len(body) < 10:
            continue

        lines = body.splitlines()
        q_lines, opt_lines, past_opts = [], [], False
        for line in lines:
            if opt_re.match(line):
                past_opts = True
            (opt_lines if past_opts else q_lines).append(line)

        question = " ".join(q_lines).strip()
        # Remove trailing answer clue accidentally grabbed into question
        question = re.sub(r'\s*(answer|ans)\s*:.*$', '', question, flags=re.IGNORECASE).strip()
        if not question or len(question) < 5:
            continue

        options = [
            f"{m.group(1).upper()}) {m.group(2).strip()}"
            for m in opt_re.finditer("\n".join(opt_lines))
        ]
        ans_m = ans_re.search(body)
        exp_m = exp_re.search(body)

        questions.append({
            "question": question,
            "options" : options,
            "correct" : ans_m.group(1).upper() if ans_m else None,
            "expl"    : textwrap.shorten(exp_m.group(1).strip(), 380, placeholder="…") if exp_m else None,
        })

    return questions


# ════════════════════════════════════════════════════════
# TFIDF SEARCH ENGINE  (fast, reliable, no GPU needed)
# ════════════════════════════════════════════════════════

def build_tfidf(questions: list):
    """
    Build a TF-IDF vectorizer + matrix from all questions.
    Stored in session_state so it persists across reruns.
    """
    corpus  = [q["question"] + " " + " ".join(q["options"]) for q in questions]
    vec     = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words=None)
    matrix  = vec.fit_transform(corpus)
    return vec, matrix


def tfidf_search(query: str, top_k: int = 5) -> list:
    """
    Search questions using TF-IDF cosine similarity.
    Falls back to keyword overlap if vectorizer not ready.
    """
    questions = st.session_state.questions
    vec       = st.session_state.tfidf_vec
    matrix    = st.session_state.tfidf_mat

    if not questions:
        return []

    # TF-IDF path
    if vec is not None and matrix is not None:
        try:
            q_vec   = vec.transform([query])
            scores  = cosine_similarity(q_vec, matrix).flatten()
            top_ids = scores.argsort()[::-1][:top_k]
            results = []
            for i in top_ids:
                s = float(scores[i])
                if s < 0.01:
                    break
                entry = dict(questions[i])
                entry["similarity"] = round(s, 3)
                results.append(entry)
            if results:
                return results
        except Exception:
            pass

    # Keyword fallback
    q_words = set(re.findall(r'\w+', query.lower()))
    scored  = []
    for q in questions:
        text  = (q["question"] + " " + " ".join(q["options"])).lower()
        words = set(re.findall(r'\w+', text))
        score = len(q_words & words) * 2 + sum(1 for w in q_words if w in text)
        if score > 0:
            scored.append((score, q))
    scored.sort(key=lambda x: -x[0])
    mx = scored[0][0] if scored else 1
    return [{**q, "similarity": round(s / mx, 3)} for s, q in scored[:top_k]]


# ════════════════════════════════════════════════════════
# SESSION STATE  — initialise once
# ════════════════════════════════════════════════════════

DEFAULTS = {
    "questions"  : [],        # list of parsed MCQ dicts
    "tfidf_vec"  : None,      # TfidfVectorizer
    "tfidf_mat"  : None,      # sparse matrix
    "selected_q" : None,      # currently viewed question dict
    "tab"        : "search",  # "search" | "upload"
    "n_files"    : 0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════
# BOTTOM TAB BAR  (decorative — real switching via buttons)
# ════════════════════════════════════════════════════════

is_search = st.session_state.tab == "search"
is_upload = st.session_state.tab == "upload"

st.markdown(f"""
<div class="bottom-bar">
  <div class="tab-btn {'tab-on' if is_search else 'tab-off'}">
    <span class="tab-icon">🔍</span>
    <span class="tab-lbl">SEARCH</span>
  </div>
  <div class="tab-btn {'tab-on' if is_upload else 'tab-off'}">
    <span class="tab-icon">📁</span>
    <span class="tab-lbl">FILES</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Real tab-switching buttons (styled as bottom bar above)
c1, c2 = st.columns(2)
with c1:
    if st.button("🔍  Search", key="nav_search", use_container_width=True):
        st.session_state.tab        = "search"
        st.session_state.selected_q = None
        st.rerun()
with c2:
    if st.button("📁  Files",  key="nav_upload", use_container_width=True):
        st.session_state.tab        = "upload"
        st.session_state.selected_q = None
        st.rerun()


# ════════════════════════════════════════════════════════
#  SEARCH TAB
# ════════════════════════════════════════════════════════

if st.session_state.tab == "search":

    n_q   = len(st.session_state.questions)
    badge = f"{n_q} Questions" if n_q else "No files yet"

    # ── Header ─────────────────────────────────────
    st.markdown(f"""
    <div class="app-header">
      <div class="hdr-row">
        <div class="app-logo">MCQ <span>Search</span> 🎯</div>
        <div class="q-badge">{badge}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search input ────────────────────────────────
    query = st.text_input(
        "search_input",
        placeholder="🔍  Type any keyword to search…",
        label_visibility="collapsed",
        key="search_box",
    )

    # ════════════════════════════════════════════════
    # VIEW A: ANSWER SCREEN
    # ════════════════════════════════════════════════
    if st.session_state.selected_q:
        q = st.session_state.selected_q

        st.markdown('<div class="ans-wrap">', unsafe_allow_html=True)

        # Back button
        st.markdown('<div class="back-wrap">', unsafe_allow_html=True)
        if st.button("← Back to results", key="back_btn"):
            st.session_state.selected_q = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Question card
        st.markdown(f"""
        <div class="q-card">
          <div class="q-card-eyebrow">❓ Question</div>
          <div class="q-card-text">{q['question']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Options
        if q["options"]:
            st.markdown('<div class="opts-section">', unsafe_allow_html=True)
            for opt in q["options"]:
                if not opt:
                    continue
                letter   = opt[0].upper()
                correct  = letter == (q.get("correct") or "")
                opt_text = opt[3:].strip() if len(opt) > 3 else opt

                row_cls = "opt-correct" if correct else ""
                ltr_cls = "opt-letter-c" if correct else ""
                txt_cls = "opt-text-c"   if correct else ""
                check   = '<span class="opt-check">✓</span>' if correct else ""

                st.markdown(f"""
                <div class="opt-row {row_cls}">
                  <div class="opt-letter {ltr_cls}">{letter}</div>
                  <div class="opt-text {txt_cls}">{opt_text}</div>
                  {check}
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        elif q.get("correct"):
            st.markdown(f"""
            <div class="ans-only-box">
              <div>
                <div class="ans-only-label">Correct Answer</div>
                <div class="ans-only-val">{q['correct']}</div>
              </div>
              <span style="font-size:32px">✅</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown('<div class="no-opt-msg">ℹ️ No answer options found in the file.</div>',
                        unsafe_allow_html=True)

        # Explanation
        if q.get("expl"):
            st.markdown(f"""
            <div class="expl-box">
              <div class="expl-eyebrow">💡 Explanation</div>
              <div class="expl-text">{q['expl']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # ans-wrap

    # ════════════════════════════════════════════════
    # VIEW B: SUGGESTION LIST
    # ════════════════════════════════════════════════
    elif query and query.strip():
        if not st.session_state.questions:
            st.markdown("""
            <div class="no-res-wrap">
              <div class="no-res-icon">📂</div>
              <div class="no-res-title">No Files Loaded</div>
              <div class="no-res-sub">Go to <b>📁 Files</b> tab and upload your MCQ files first.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            results = tfidf_search(query.strip(), top_k=5)

            if not results:
                st.markdown(f"""
                <div class="no-res-wrap">
                  <div class="no-res-icon">😕</div>
                  <div class="no-res-title">No matches for "{query}"</div>
                  <div class="no-res-sub">Try a shorter or different keyword.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="sugg-wrap">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="sugg-label">🔎 {len(results)} suggestions</div>',
                    unsafe_allow_html=True,
                )

                for i, res in enumerate(results):
                    short_q = textwrap.shorten(res["question"], 100, placeholder="…")
                    sim_pct = int(res["similarity"] * 100)
                    has_ans = res.get("correct") is not None
                    n_opts  = len(res.get("options", []))

                    # Clickable suggestion
                    if st.button(short_q, key=f"res_{i}_{hash(res['question'])}", use_container_width=True):
                        st.session_state.selected_q = res
                        st.rerun()

                    # Meta badges
                    ans_badge = (
                        '<span class="badge badge-ok">✓ Has Answer</span>'
                        if has_ans else
                        '<span class="badge badge-none">No Answer</span>'
                    )
                    st.markdown(f"""
                    <div class="meta-row">
                      <span class="badge badge-match">{sim_pct}% match</span>
                      {ans_badge}
                      <span class="badge badge-opts">{n_opts} options</span>
                    </div>
                    <div class="sugg-divider"></div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)  # sugg-wrap

    # ════════════════════════════════════════════════
    # VIEW C: EMPTY STATE
    # ════════════════════════════════════════════════
    else:
        n_q = len(st.session_state.questions)
        if n_q:
            title = "Ready! Start Typing 🚀"
            msg   = f"<b>{n_q} questions</b> loaded.<br>Type any keyword above to search instantly."
        else:
            title = "Welcome 👋"
            msg   = "Go to <b>📁 Files</b> tab below,<br>upload your MCQ files, then search here."

        st.markdown(f"""
        <div class="empty-wrap">
          <div class="empty-orb">🎯</div>
          <div class="empty-title">{title}</div>
          <div class="empty-sub">{msg}</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  UPLOAD TAB
# ════════════════════════════════════════════════════════

elif st.session_state.tab == "upload":

    # Header
    st.markdown("""
    <div class="upload-hdr">
      <div class="upload-hdr-title">📁 <span>Upload Files</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)

    # Stats
    n_q    = len(st.session_state.questions)
    n_f    = st.session_state.n_files
    idx_ok = st.session_state.tfidf_vec is not None

    st.markdown(f"""
    <div class="stat-strip">
      <div class="stat-box">
        <div class="stat-num">{n_q}</div>
        <div class="stat-lbl">Questions</div>
      </div>
      <div class="stat-box">
        <div class="stat-num">{n_f}</div>
        <div class="stat-lbl">Files</div>
      </div>
      <div class="stat-box">
        <div class="stat-num">{'✓' if idx_ok else '—'}</div>
        <div class="stat-lbl">Index</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # MCQ file uploader
    st.markdown('<div class="upload-sec-label">📝 MCQ Question Files (PDF or TXT)</div>',
                unsafe_allow_html=True)
    mcq_files = st.file_uploader(
        "mcq_files", type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="mcq_up",
    )

    # Notes uploader (optional)
    st.markdown('<div class="upload-sec-label">📄 Study Notes PDFs (optional)</div>',
                unsafe_allow_html=True)
    note_files = st.file_uploader(
        "note_files", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="notes_up",
    )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # Process button
    st.markdown('<div class="proc-btn">', unsafe_allow_html=True)
    process = st.button("⚡  Process & Build Index", use_container_width=True, key="proc_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    if process:
        all_files = list(mcq_files or []) + list(note_files or [])
        if not all_files:
            st.warning("⚠️ Please upload at least one file first.")
        else:
            all_q = []
            prog  = st.progress(0, text="Reading files…")

            for fi, f in enumerate(all_files):
                prog.progress((fi + 1) / len(all_files), text=f"Reading {f.name}…")
                raw = (
                    extract_pdf(f.read()) if f.name.lower().endswith(".pdf")
                    else extract_txt(f.read())
                )
                parsed = parse_mcq(raw)

                # Aggressive fallback: treat every sentence as a question
                if not parsed:
                    for sent in re.split(r'(?<=[.?!])\s+', raw):
                        sent = sent.strip()
                        if 25 < len(sent) < 500:
                            parsed.append({
                                "question": sent,
                                "options" : [],
                                "correct" : None,
                                "expl"    : None,
                            })
                all_q.extend(parsed)

            prog.progress(1.0, text="Building search index…")
            st.session_state.questions = all_q
            st.session_state.n_files   = len(all_files)

            if all_q:
                vec, mat = build_tfidf(all_q)
                st.session_state.tfidf_vec = vec
                st.session_state.tfidf_mat = mat
                prog.empty()
                st.success(f"✅ {len(all_q)} questions indexed from {len(all_files)} file(s)! Go to 🔍 Search tab.")
                st.balloons()
            else:
                prog.empty()
                st.error("❌ Could not detect any questions. Check your file format.")

    st.markdown('</div>', unsafe_allow_html=True)  # upload-wrap
