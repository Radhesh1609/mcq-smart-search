"""
MCQ Smart Search – Mobile Phone UI
====================================
Streamlit app with phone-frame layout.
Flow: Upload MCQ files (PDF/TXT) → type to search → tap a question → see the answer.

Install & run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import re
import io
import textwrap

import numpy as np
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  ← must be the very first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MCQ Smart Search",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

/* ── reset ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: #080d14; color: #e2e8f0; }

/* ── hide Streamlit chrome ──────────────── */
#MainMenu, footer, header { visibility: hidden; height: 0; }
.block-container { padding: 0 !important; max-width: 430px !important; margin: 0 auto !important; }
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }

/* ── app wrapper ────────────────────────── */
.app-wrap {
    max-width: 430px;
    margin: 0 auto;
    min-height: 100vh;
    background: #0d1117;
    display: flex;
    flex-direction: column;
}

/* ── status bar ─────────────────────────── */
.status-bar {
    background: #060a10;
    padding: 10px 22px 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.status-bar span { color: #94a3b8; font-size: 11px; font-weight: 500; letter-spacing: 0.3px; }

/* ── app header ─────────────────────────── */
.app-header {
    background: linear-gradient(170deg, #060a10 0%, #0c1a2e 60%, #0f2340 100%);
    padding: 14px 20px 16px;
    border-bottom: 1px solid rgba(56,189,248,0.1);
}
.app-eyebrow {
    color: #38bdf8;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.app-title {
    color: #f0f9ff;
    font-size: 19px;
    font-weight: 600;
    margin-bottom: 14px;
    letter-spacing: -0.3px;
}

/* ── search bar in header ───────────────── */
.stTextInput > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(56,189,248,0.22) !important;
    border-radius: 13px !important;
    padding: 0 14px !important;
    transition: border-color 0.2s, background 0.2s;
}
.stTextInput > div > div:focus-within {
    border-color: rgba(56,189,248,0.5) !important;
    background: rgba(255,255,255,0.10) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.08) !important;
}
.stTextInput input {
    color: #f0f9ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    caret-color: #38bdf8 !important;
    padding: 11px 0 !important;
}
.stTextInput input::placeholder { color: #475569 !important; }

/* ── nav tab buttons ────────────────────── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: transparent !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 7px 8px !important;
    transition: all 0.15s !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
    background: rgba(56,189,248,0.06) !important;
}

/* ── content area ───────────────────────── */
.content-area {
    flex: 1;
    background: #0d1117;
    padding: 14px 16px 80px;
    min-height: 500px;
}

/* ── section label ──────────────────────── */
.sec-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    color: #334155;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-left: 2px;
}

/* ── suggestion result card ─────────────── */
.sug-card {
    background: #131c2b;
    border: 1px solid #1a2d45;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: background 0.15s, border-color 0.15s, transform 0.12s;
}
.sug-card:hover {
    background: #1a2d45;
    border-color: #2563eb;
    transform: translateX(3px);
}
.sug-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.2);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 11px; font-weight: 600;
    color: #38bdf8;
    font-family: 'DM Mono', monospace;
}
.sug-q {
    font-size: 13px; font-weight: 500;
    color: #cbd5e1; line-height: 1.45;
    margin-bottom: 2px;
}
.sug-meta {
    font-size: 11px; color: #475569;
}
.sug-chevron { color: #1e3a5f; margin-left: auto; flex-shrink: 0; font-size: 13px; }

/* ── invisible overlay button on cards ──── */
.sug-btn-wrap { position: relative; margin-top: -55px; opacity: 0; height: 55px; }
.sug-btn-wrap .stButton > button {
    height: 55px !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    cursor: pointer !important;
}

/* ── answer back button ─────────────────── */
.stButton > button[kind="secondary"] {
    background: rgba(56,189,248,0.07) !important;
    border: 1px solid rgba(56,189,248,0.18) !important;
    border-radius: 10px !important;
    color: #38bdf8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.15s !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(56,189,248,0.13) !important;
}

/* ── answer card ────────────────────────── */
.ans-header {
    background: linear-gradient(145deg, #060a10 0%, #0c1a2e 100%);
    border: 1px solid #1a3050;
    border-bottom: none;
    border-radius: 16px 16px 0 0;
    padding: 16px 18px 18px;
}
.ans-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.22);
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 10px; font-weight: 600;
    color: #38bdf8; letter-spacing: 0.5px;
    margin-bottom: 10px;
}
.ans-q-text {
    color: #f0f9ff;
    font-size: 15px; font-weight: 600;
    line-height: 1.55;
}
.ans-body {
    background: #111827;
    border: 1px solid #1a3050;
    border-top: none;
    border-radius: 0 0 16px 16px;
    padding: 14px 16px 16px;
}
.opts-label {
    font-size: 10px; font-weight: 600;
    color: #334155; letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.opt-row {
    display: flex; align-items: center;
    gap: 10px; padding: 10px 12px;
    border-radius: 10px; margin-bottom: 6px;
    border: 1px solid transparent;
}
.opt-correct {
    background: rgba(22,163,74,0.1);
    border-color: rgba(22,163,74,0.32);
}
.opt-wrong {
    background: rgba(255,255,255,0.025);
    border-color: rgba(255,255,255,0.06);
}
.opt-letter {
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; flex-shrink: 0;
    font-family: 'DM Mono', monospace;
}
.opt-ltr-c { background: #16a34a; color: #fff; }
.opt-ltr-w { background: rgba(255,255,255,0.05); color: #475569; border: 1px solid rgba(255,255,255,0.08); }
.opt-txt-c { font-size: 13px; color: #86efac; font-weight: 500; flex: 1; line-height: 1.4; }
.opt-txt-w { font-size: 13px; color: #64748b; flex: 1; line-height: 1.4; }
.opt-tick  { color: #4ade80; font-size: 14px; margin-left: auto; flex-shrink: 0; }

.expl-box {
    background: rgba(245,158,11,0.07);
    border-left: 3px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 11px 14px;
    margin-top: 12px;
}
.expl-title {
    font-size: 10px; font-weight: 600;
    letter-spacing: 1px; color: #d97706;
    text-transform: uppercase; margin-bottom: 5px;
}
.expl-text { font-size: 13px; color: #fde68a; line-height: 1.6; }

/* ── no-answer fallback ─────────────────── */
.no-opts { font-size: 13px; color: #475569; font-style: italic; padding: 6px 0; }

/* ── empty state ────────────────────────── */
.empty-wrap {
    display: flex; flex-direction: column;
    align-items: center; text-align: center;
    padding: 50px 20px 30px;
}
.empty-icon-wrap {
    width: 68px; height: 68px; border-radius: 50%;
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.15);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px; margin-bottom: 18px;
}
.empty-title { font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 8px; }
.empty-sub   { font-size: 13px; color: #475569; line-height: 1.65; max-width: 280px; margin-bottom: 26px; }
.chip-row    { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.chip {
    background: #131c2b; border: 1px solid #1a2d45;
    border-radius: 20px; padding: 6px 14px;
    font-size: 12px; color: #7dd3fc; cursor: default;
}

/* ── no results ─────────────────────────── */
.no-results {
    text-align: center; padding: 40px 20px;
    color: #475569; font-size: 14px; line-height: 1.6;
}

/* ── upload panel ───────────────────────── */
.stats-strip { display: flex; gap: 10px; margin-bottom: 16px; }
.stat-pill {
    flex: 1; background: #131c2b;
    border: 1px solid #1a2d45; border-radius: 12px;
    padding: 12px; text-align: center;
}
.stat-num { font-size: 24px; font-weight: 600; color: #38bdf8; font-family: 'DM Mono', monospace; }
.stat-lbl { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── process button ─────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 20px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button[kind="primary"]:hover { opacity: 0.87 !important; }

/* ── file uploader ──────────────────────── */
[data-testid="stFileUploader"] {
    background: #131c2b !important;
    border: 1px dashed #1a2d45 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    margin-bottom: 12px !important;
}
[data-testid="stFileUploader"] label { color: #64748b !important; font-size: 13px !important; }
[data-testid="stFileUploader"] small { color: #334155 !important; }

/* ── browse table ───────────────────────── */
.browse-row {
    background: #131c2b; border: 1px solid #1a2d45;
    border-radius: 12px; padding: 11px 14px;
    margin-bottom: 7px; display: flex;
    align-items: flex-start; gap: 12px;
}
.browse-num {
    font-size: 11px; font-weight: 600;
    color: #334155; font-family: 'DM Mono', monospace;
    min-width: 26px; padding-top: 1px;
}
.browse-q   { font-size: 13px; color: #94a3b8; line-height: 1.45; flex: 1; }
.browse-ans {
    font-size: 11px; font-weight: 600;
    color: #4ade80; font-family: 'DM Mono', monospace;
    flex-shrink: 0; background: rgba(22,163,74,0.1);
    border: 1px solid rgba(22,163,74,0.22);
    border-radius: 6px; padding: 2px 8px;
    margin-top: 1px;
}

/* ── bottom nav ─────────────────────────── */
.bottom-nav {
    position: fixed; bottom: 0; left: 50%;
    transform: translateX(-50%);
    width: 100%; max-width: 430px;
    background: rgba(8,13,20,0.97);
    backdrop-filter: blur(12px);
    border-top: 1px solid #1a2d45;
    padding: 10px 0 18px;
    display: flex; justify-content: space-around;
    z-index: 999;
}
.nav-item { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.nav-icon  { font-size: 19px; }
.nav-lbl   { font-size: 10px; font-weight: 500; letter-spacing: 0.3px; }
.nav-on  .nav-lbl { color: #38bdf8; }
.nav-off .nav-lbl { color: #334155; }

/* ── streamlit info/success/warning ─────── */
.stAlert { border-radius: 12px !important; font-size: 13px !important; }

/* ── spinner colour ─────────────────────── */
.stSpinner > div { border-top-color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5
MIN_SIM     = 0.20


# ──────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf(data: bytes) -> str:
    parts = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
    except Exception:
        pass
    return "\n".join(parts)


def extract_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


# ──────────────────────────────────────────────────────────────────────────────
# MCQ PARSER
# ──────────────────────────────────────────────────────────────────────────────

def parse_mcq(raw: str) -> list[dict]:
    """Parse numbered MCQ blocks from raw text. Returns list of question dicts."""
    questions = []

    # Try numbered blocks first (1. / Q1. / Question 1:)
    block_re = re.compile(
        r'(?:^|\n)\s*(?:Q(?:uestion)?\s*)?(\d+)[.):\s]+(.+?)'
        r'(?=\n\s*(?:Q(?:uestion)?\s*)?\d+[.):\s]+|\Z)',
        re.DOTALL | re.IGNORECASE,
    )
    blocks = block_re.findall(raw)

    # Fallback: split by blank lines
    if len(blocks) < 3:
        paras  = re.split(r'\n{2,}', raw.strip())
        blocks = [(str(i + 1), p) for i, p in enumerate(paras) if len(p.strip()) > 30]

    opt_re  = re.compile(r'^\s*[\(\[]?([A-Ea-e])[\).\]\s]+(.+)', re.MULTILINE)
    ans_re  = re.compile(
        r'(?:answer|ans|correct(?:\s+answer)?|key)\s*[:\-–]\s*\(?([A-Ea-e])\)?',
        re.IGNORECASE,
    )
    expl_re = re.compile(
        r'(?:explanation|reason|note|hint)\s*[:\-–]\s*(.+)',
        re.IGNORECASE | re.DOTALL,
    )

    for _, block in blocks:
        block = block.strip()
        if not block or len(block) < 10:
            continue

        # Split question text from option lines
        lines               = block.split("\n")
        q_lines, opt_lines  = [], []
        in_opts             = False
        for line in lines:
            if opt_re.match(line):
                in_opts = True
            (opt_lines if in_opts else q_lines).append(line)

        question_text = " ".join(q_lines).strip()
        if not question_text:
            continue

        options = [
            f"{m.group(1).upper()}) {m.group(2).strip()}"
            for m in opt_re.finditer("\n".join(opt_lines))
        ]

        ans_m  = ans_re.search(block)
        expl_m = expl_re.search(block)

        questions.append({
            "question"   : question_text,
            "options"    : options,
            "correct_key": ans_m.group(1).upper() if ans_m else None,
            "explanation": expl_m.group(1).strip()[:500] if expl_m else None,
            "raw_text"   : block,
        })

    return questions


# ──────────────────────────────────────────────────────────────────────────────
# SEMANTIC SEARCH  (sentence-transformers + FAISS)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    """Download and cache the embedding model once per session."""
    return SentenceTransformer(EMBED_MODEL)


def build_index(questions: list[dict], model: SentenceTransformer):
    """Encode questions and build a FAISS inner-product index."""
    corpus = [
        q["question"] + " " + " ".join(q["options"])
        for q in questions
    ]
    embs = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    embs = np.array(embs, dtype="float32")
    idx  = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return idx


def semantic_search(query: str, model: SentenceTransformer,
                    index, questions: list[dict]) -> list[dict]:
    """Return top-K matching questions for the query."""
    if not query.strip() or index is None:
        return []
    q_vec = np.array(
        model.encode([query], normalize_embeddings=True),
        dtype="float32",
    )
    scores, ids = index.search(q_vec, min(TOP_K, index.ntotal))
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1 or score < MIN_SIM:
            continue
        entry = dict(questions[idx])
        entry["similarity"] = float(score)
        out.append(entry)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "questions"   : [],      # list[dict] – all parsed MCQ records
    "faiss_index" : None,    # FAISS index
    "results"     : [],      # current search results
    "selected"    : None,    # question dict being displayed
    "view"        : "home",  # "home" | "search" | "answer" | "upload" | "browse"
    "last_query"  : "",      # for change detection
    "file_count"  : 0,
    "nav_active"  : "search",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ──────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _status_bar():
    st.markdown("""
    <div class="status-bar">
      <span style="font-family:'DM Mono',monospace">9:41</span>
      <span>● ● ●&nbsp;&nbsp;WiFi&nbsp;&nbsp;100%</span>
    </div>""", unsafe_allow_html=True)


def _app_header():
    st.markdown("""
    <div class="app-header">
      <div class="app-eyebrow">Study Tool</div>
      <div class="app-title">MCQ Smart Search 🧠</div>
    </div>""", unsafe_allow_html=True)


def _bottom_nav():
    icons = {"search": "🔍", "browse": "📋", "upload": "📁"}
    labels = {"search": "Search", "browse": "Browse", "upload": "Files"}
    active = st.session_state.nav_active
    items  = "".join(
        f'<div class="nav-item {"nav-on" if k == active else "nav-off"}">'
        f'<span class="nav-icon">{v}</span>'
        f'<span class="nav-lbl">{labels[k]}</span></div>'
        for k, v in icons.items()
    )
    st.markdown(f'<div class="bottom-nav">{items}</div>', unsafe_allow_html=True)


def _nav_buttons():
    """Render three nav tab buttons that switch views."""
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔍 Search", use_container_width=True, key="nav_search"):
            st.session_state.view       = "home"
            st.session_state.nav_active = "search"
            st.session_state.selected   = None
            st.rerun()
    with c2:
        if st.button("📋 Browse", use_container_width=True, key="nav_browse"):
            st.session_state.view       = "browse"
            st.session_state.nav_active = "browse"
            st.session_state.selected   = None
            st.rerun()
    with c3:
        if st.button("📁 Files", use_container_width=True, key="nav_files"):
            st.session_state.view       = "upload"
            st.session_state.nav_active = "upload"
            st.session_state.selected   = None
            st.rerun()


# ─── EMPTY / HOME STATE ──────────────────────────────────────────────────────

def render_home():
    loaded = len(st.session_state.questions)
    if loaded:
        sub = f"✅ {loaded} questions loaded and ready.<br>Start typing in the search bar above."
    else:
        sub = "Upload your MCQ files using the <b>📁 Files</b> tab below,<br>then search instantly with AI."

    chips = "".join(
        f'<span class="chip">{c}</span>'
        for c in ["photosynthesis", "Newton's law", "DNA replication", "osmosis", "mitosis"]
    )
    st.markdown(f"""
    <div class="empty-wrap">
      <div class="empty-icon-wrap">🔍</div>
      <p class="empty-title">Search your questions</p>
      <p class="empty-sub">{sub}</p>
      <p class="sec-label" style="width:100%;text-align:center;margin-bottom:10px">Try these topics</p>
      <div class="chip-row">{chips}</div>
    </div>""", unsafe_allow_html=True)


# ─── SUGGESTION LIST ─────────────────────────────────────────────────────────

def render_suggestions():
    results = st.session_state.results

    if not st.session_state.questions:
        st.markdown("""
        <div class="no-results">
          📂 No questions loaded yet.<br>
          Go to <b>📁 Files</b> to upload your MCQ files.
        </div>""", unsafe_allow_html=True)
        return

    if not results:
        st.markdown("""
        <div class="no-results">
          😕 No matching questions found.<br>
          Try different or shorter keywords.
        </div>""", unsafe_allow_html=True)
        return

    st.markdown(f'<p class="sec-label">Top {len(results)} matches</p>', unsafe_allow_html=True)

    for i, res in enumerate(results):
        short_q = textwrap.shorten(res["question"], 85, placeholder="…")
        pct     = int(res["similarity"] * 100)
        letter  = chr(65 + i)

        st.markdown(f"""
        <div class="sug-card">
          <div class="sug-avatar">{letter}</div>
          <div style="flex:1;min-width:0">
            <p class="sug-q">{short_q}</p>
            <p class="sug-meta">Relevance: {pct}%&nbsp;&nbsp;
              {"✅ Answer available" if res.get("correct_key") else "❓ No answer key"}</p>
          </div>
          <span class="sug-chevron">›</span>
        </div>""", unsafe_allow_html=True)

        if st.button(f"Open question {i+1}", key=f"tap_{i}",
                     use_container_width=True, help=res["question"]):
            st.session_state.selected = res
            st.session_state.view     = "answer"
            st.rerun()

        # pull the plain button UP over the card using negative margin
        st.markdown("""
        <style>
        div[data-testid="stButton"]:last-child > button {
            margin-top: -52px !important;
            opacity: 0 !important;
            height: 52px !important;
            cursor: pointer !important;
        }
        </style>""", unsafe_allow_html=True)


# ─── ANSWER VIEW ─────────────────────────────────────────────────────────────

def render_answer():
    q = st.session_state.selected
    if not q:
        st.session_state.view = "home"
        st.rerun()

    if st.button("← Back to results", key="back_btn"):
        st.session_state.view     = "search"
        st.session_state.selected = None
        st.rerun()

    pct = int(q.get("similarity", 0) * 100)

    # ── Question header card ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="ans-header">
      <div class="ans-badge">🎯 Relevance: {pct}%</div>
      <p class="ans-q-text">❓ {q['question']}</p>
    </div>""", unsafe_allow_html=True)

    # ── Build options HTML ────────────────────────────────────────────────────
    opts_html = ""

    if q["options"]:
        for opt in q["options"]:
            if not opt:
                continue
            letter     = opt[0].upper()
            text       = opt[3:] if len(opt) > 3 else opt
            is_correct = letter == (q.get("correct_key") or "")

            row_cls  = "opt-correct"  if is_correct else "opt-wrong"
            ltr_cls  = "opt-ltr-c"   if is_correct else "opt-ltr-w"
            txt_cls  = "opt-txt-c"   if is_correct else "opt-txt-w"
            tick     = '<span class="opt-tick">✓</span>' if is_correct else ""

            opts_html += f"""
            <div class="opt-row {row_cls}">
              <div class="opt-letter {ltr_cls}">{letter}</div>
              <span class="{txt_cls}">{text}</span>
              {tick}
            </div>"""

    elif q.get("correct_key"):
        opts_html = (
            f'<p style="color:#86efac;font-weight:600;font-size:14px;padding:8px 0">'
            f'✅ Correct Answer: &nbsp;<span style="font-family:\'DM Mono\',monospace">'
            f'{q["correct_key"]}</span></p>'
        )
    else:
        opts_html = '<p class="no-opts">ℹ️ No multiple-choice options found for this entry.</p>'

    # ── Explanation ───────────────────────────────────────────────────────────
    expl_html = ""
    if q.get("explanation"):
        expl_html = f"""
        <div class="expl-box">
          <p class="expl-title">📌 Explanation</p>
          <p class="expl-text">{q['explanation']}</p>
        </div>"""

    st.markdown(f"""
    <div class="ans-body">
      <p class="opts-label">Options</p>
      {opts_html}
      {expl_html}
    </div>""", unsafe_allow_html=True)


# ─── UPLOAD / FILES VIEW ─────────────────────────────────────────────────────

def render_upload():
    n = len(st.session_state.questions)
    f = st.session_state.file_count

    st.markdown(f"""
    <div class="stats-strip">
      <div class="stat-pill">
        <div class="stat-num">{n}</div>
        <div class="stat-lbl">Questions</div>
      </div>
      <div class="stat-pill">
        <div class="stat-num">{f}</div>
        <div class="stat-lbl">Files loaded</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sec-label">Upload MCQ Files</p>', unsafe_allow_html=True)
    mcq_files = st.file_uploader(
        "Drop PDF or TXT question files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="mcq_uploader",
        help="Supports: numbered MCQ PDFs, plain-text MCQ lists.",
    )

    st.markdown('<p class="sec-label" style="margin-top:4px">Upload Study Notes (optional)</p>',
                unsafe_allow_html=True)
    note_files = st.file_uploader(
        "Drop PDF notes for extra context",
        type=["pdf"],
        accept_multiple_files=True,
        key="notes_uploader",
        help="Notes are not searched directly but can be used for future context.",
    )

    if st.button("⚡  Process & Build AI Index", type="primary",
                 use_container_width=True, key="process_btn"):
        if not mcq_files:
            st.warning("⚠️ Please upload at least one MCQ file first.")
        else:
            with st.spinner("Reading files and building AI index — this may take a moment…"):
                all_qs = []
                for uf in mcq_files:
                    raw = (extract_pdf(uf.read())
                           if uf.name.lower().endswith(".pdf")
                           else extract_txt(uf.read()))
                    parsed = parse_mcq(raw)

                    # Last-resort fallback: treat every paragraph as a question
                    if not parsed:
                        for para in raw.split("\n\n"):
                            para = para.strip()
                            if len(para) > 20:
                                parsed.append({
                                    "question"   : para[:300],
                                    "options"    : [],
                                    "correct_key": None,
                                    "explanation": None,
                                    "raw_text"   : para,
                                })

                    all_qs.extend(parsed)

                st.session_state.questions  = all_qs
                st.session_state.file_count = len(mcq_files) + len(note_files or [])

                if all_qs:
                    model = load_model()
                    st.session_state.faiss_index = build_index(all_qs, model)
                    st.success(
                        f"✅ Indexed **{len(all_qs)} questions** from "
                        f"**{len(mcq_files)}** MCQ file(s). Ready to search!"
                    )
                else:
                    st.warning(
                        "⚠️ Could not detect any questions. "
                        "Make sure your file uses numbered questions (1. 2. 3. …) "
                        "or try a plain .txt file."
                    )


# ─── BROWSE VIEW ─────────────────────────────────────────────────────────────

def render_browse():
    questions = st.session_state.questions

    if not questions:
        st.markdown("""
        <div class="no-results">
          📂 No questions loaded yet.<br>Go to <b>📁 Files</b> to upload your MCQ files.
        </div>""", unsafe_allow_html=True)
        return

    st.markdown(f'<p class="sec-label">All {len(questions)} questions</p>',
                unsafe_allow_html=True)

    # Search filter within browse
    browse_filter = st.text_input(
        "Filter",
        placeholder="🔎  Filter questions…",
        label_visibility="collapsed",
        key="browse_filter",
    )

    filtered = questions
    if browse_filter.strip():
        kw = browse_filter.lower()
        filtered = [q for q in questions
                    if kw in q["question"].lower()
                    or kw in " ".join(q["options"]).lower()]

    for i, q in enumerate(filtered[:60]):   # limit to 60 rows for performance
        short_q = textwrap.shorten(q["question"], 100, placeholder="…")
        ans_tag = (f'<span class="browse-ans">{q["correct_key"]}</span>'
                   if q.get("correct_key") else "")
        st.markdown(f"""
        <div class="browse-row">
          <span class="browse-num">#{i+1}</span>
          <span class="browse-q">{short_q}</span>
          {ans_tag}
        </div>""", unsafe_allow_html=True)

        if st.button("", key=f"browse_tap_{i}", use_container_width=True,
                     help=q["question"]):
            q_copy = dict(q)
            q_copy.setdefault("similarity", 1.0)
            st.session_state.selected   = q_copy
            st.session_state.view       = "answer"
            st.session_state.nav_active = "search"
            st.rerun()

    if len(filtered) > 60:
        st.markdown(
            f'<p style="color:#334155;font-size:12px;text-align:center;padding:10px">'
            f'Showing 60 of {len(filtered)} — use the filter to narrow down.</p>',
            unsafe_allow_html=True,
        )
    if browse_filter and not filtered:
        st.markdown('<div class="no-results">😕 No questions match that filter.</div>',
                    unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP  –  assemble the screen
# ──────────────────────────────────────────────────────────────────────────────

_status_bar()
_app_header()

# ── Search input (always visible) ────────────────────────────────────────────
query = st.text_input(
    "search",
    placeholder="🔍  Type a keyword to find questions…",
    label_visibility="collapsed",
    key="main_search",
)

# ── Nav tabs ──────────────────────────────────────────────────────────────────
_nav_buttons()

# ── Handle search ─────────────────────────────────────────────────────────────
if query and query != st.session_state.last_query:
    st.session_state.last_query = query
    st.session_state.nav_active = "search"
    if st.session_state.faiss_index and st.session_state.questions:
        model = load_model()
        st.session_state.results = semantic_search(
            query, model,
            st.session_state.faiss_index,
            st.session_state.questions,
        )
    else:
        st.session_state.results = []
    if st.session_state.view != "answer":
        st.session_state.view = "search"

if not query and st.session_state.view == "search":
    st.session_state.view = "home"

# ── Route to the right view ───────────────────────────────────────────────────
st.markdown('<div class="content-area">', unsafe_allow_html=True)

view = st.session_state.view

if view == "home":
    render_home()
elif view == "search":
    render_suggestions()
elif view == "answer":
    render_answer()
elif view == "upload":
    render_upload()
elif view == "browse":
    render_browse()
else:
    render_home()

st.markdown('</div>', unsafe_allow_html=True)

_bottom_nav()
