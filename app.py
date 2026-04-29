"""
MCQ Smart Search Application
=============================
A Streamlit app that lets you upload study PDFs and MCQ question files,
then intelligently search for questions and answers using semantic AI embeddings.

Run with:
    streamlit run app.py
"""

import re
import io
import time
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MCQ Smart Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  –  clean, modern look
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #7dd3fc !important; }

/* ---------- main header ---------- */
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0ea5e9 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0; }

/* ---------- section cards ---------- */
.section-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e3a5f;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 2px solid #e0f2fe;
    padding-bottom: 0.5rem;
}

/* ---------- suggestion pill ---------- */
.suggestion-item {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.35rem 0;
    cursor: pointer;
    font-size: 0.92rem;
    color: #0369a1;
    transition: all 0.18s;
}
.suggestion-item:hover {
    background: #0ea5e9;
    color: white;
    border-color: #0ea5e9;
    transform: translateX(4px);
}

/* ---------- answer card ---------- */
.answer-card {
    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
    border: 1px solid #86efac;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-top: 1rem;
}
.question-text {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1.2rem;
    line-height: 1.6;
}
.option-row {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.55rem 0.9rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    font-size: 0.95rem;
    color: #334155;
    border: 1px solid transparent;
}
.option-correct {
    background: #dcfce7;
    border-color: #4ade80;
    color: #166534;
    font-weight: 600;
}
.option-wrong {
    background: #f8fafc;
    border-color: #e2e8f0;
}
.correct-badge {
    display: inline-block;
    background: #16a34a;
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    margin-left: 0.5rem;
    vertical-align: middle;
}
.explanation-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.1rem;
    margin-top: 1.2rem;
    font-size: 0.92rem;
    color: #78350f;
    line-height: 1.6;
}
.similarity-badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

/* ---------- stats bar ---------- */
.stats-box {
    background: linear-gradient(135deg, #1e3a5f, #0ea5e9);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: white;
    text-align: center;
}
.stats-num { font-size: 1.8rem; font-weight: 700; }
.stats-label { font-size: 0.8rem; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, accurate
TOP_K_RESULTS        = 5
MIN_SIMILARITY       = 0.20                  # cosine similarity threshold


# ═════════════════════════════════════════════
# HELPER – PDF & TEXT EXTRACTION
# ═════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF using pdfplumber.
    Falls back to page-by-page extraction if full-doc fails.
    """
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        st.warning(f"PDF read issue (partial text may be returned): {e}")
    return "\n".join(text_parts)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode a plain-text file, trying UTF-8 then latin-1 as fallback."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")


# ═════════════════════════════════════════════
# HELPER – MCQ PARSER
# ═════════════════════════════════════════════

def parse_mcq_blocks(raw_text: str) -> list[dict]:
    """
    Parse MCQ blocks from raw text.

    Handles common question formats:
      1. Question text
         A) ...  or  a) ...  or  A. ...  or  (A) ...
         Answer: A  / Correct: B  / Ans: C

    Returns a list of dicts:
        {
          "question"    : str,
          "options"     : list[str],          # ["A) ...", "B) ..."]
          "correct_key" : str | None,         # "A", "B", "C", "D"
          "explanation" : str | None,
          "raw_text"    : str,                # full block (for search indexing)
        }
    """
    questions = []

    # ── Split into numbered blocks (1. / Q1. / Question 1:)
    block_pattern = re.compile(
        r'(?:^|\n)\s*(?:Q(?:uestion)?\s*)?(\d+)[.):\s]+(.+?)(?=\n\s*(?:Q(?:uestion)?\s*)?\d+[.):\s]+|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    blocks = block_pattern.findall(raw_text)

    # ── Also try splitting by blank lines if numbered blocks are few
    if len(blocks) < 3:
        paragraphs = re.split(r'\n{2,}', raw_text.strip())
        blocks = [(str(i+1), p) for i, p in enumerate(paragraphs) if len(p.strip()) > 30]

    option_pattern = re.compile(
        r'^\s*[\(\[]?([A-Ea-e])[\).\]\s]+(.+)', re.MULTILINE
    )
    answer_pattern = re.compile(
        r'(?:answer|ans|correct(?:\s+answer)?|key)\s*[:\-–]\s*\(?([A-Ea-e])\)?',
        re.IGNORECASE
    )
    explanation_pattern = re.compile(
        r'(?:explanation|reason|note|hint)\s*[:\-–]\s*(.+)',
        re.IGNORECASE | re.DOTALL
    )

    for _, block_text in blocks:
        block_text = block_text.strip()
        if not block_text or len(block_text) < 10:
            continue

        lines = block_text.split("\n")

        # First line(s) before options = question text
        q_lines, opt_lines = [], []
        in_options = False
        for line in lines:
            if option_pattern.match(line):
                in_options = True
            if in_options:
                opt_lines.append(line)
            else:
                q_lines.append(line)

        question_text = " ".join(q_lines).strip()
        if not question_text:
            continue

        # Parse options
        options = []
        for m in option_pattern.finditer("\n".join(opt_lines)):
            label = m.group(1).upper()
            text  = m.group(2).strip()
            options.append(f"{label}) {text}")

        # Find correct answer
        answer_match = answer_pattern.search(block_text)
        correct_key  = answer_match.group(1).upper() if answer_match else None

        # Find explanation
        exp_match   = explanation_pattern.search(block_text)
        explanation = exp_match.group(1).strip()[:400] if exp_match else None

        questions.append({
            "question"   : question_text,
            "options"    : options,
            "correct_key": correct_key,
            "explanation": explanation,
            "raw_text"   : block_text,
        })

    return questions


# ═════════════════════════════════════════════
# HELPER – NOTE CONTEXT EXTRACTOR
# ═════════════════════════════════════════════

def find_context_in_notes(query: str, notes_text: str, window: int = 500) -> Optional[str]:
    """
    Find the most relevant snippet from notes text for a given query.
    Simple sliding-window keyword match (fast, no embedding overhead).
    """
    if not notes_text:
        return None

    query_words = set(re.findall(r'\w+', query.lower()))
    best_score, best_snippet = 0, None
    words     = notes_text.split()
    step      = max(1, window // 10)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + window])
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        overlap     = len(query_words & chunk_words)
        if overlap > best_score:
            best_score   = overlap
            best_snippet = chunk

    return textwrap.shorten(best_snippet, width=600, placeholder=" …") if best_snippet else None


# ═════════════════════════════════════════════
# SEMANTIC SEARCH ENGINE  (FAISS + Sentence Transformers)
# ═════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    """Load (and cache) the sentence-transformer model once per session."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_faiss_index(questions: list[dict], model: SentenceTransformer):
    """
    Build a FAISS flat inner-product index from question embeddings.

    We encode the full question text (+ option text) for richer matching.
    Returns (index, embeddings_matrix).
    """
    corpus = []
    for q in questions:
        text = q["question"]
        if q["options"]:
            text += " " + " ".join(q["options"])
        corpus.append(text)

    embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)           # Inner Product ≈ cosine sim (normalised vecs)
    index.add(embeddings)
    return index, embeddings


def semantic_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    questions: list[dict],
    top_k: int = TOP_K_RESULTS,
) -> list[dict]:
    """
    Encode query and retrieve the top-k most similar questions from the FAISS index.
    Returns enriched question dicts with an added 'similarity' score.
    """
    if not query.strip() or index is None:
        return []

    q_vec = model.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")

    scores, indices = index.search(q_vec, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < MIN_SIMILARITY:
            continue
        entry = dict(questions[idx])
        entry["similarity"] = float(score)
        results.append(entry)

    return results


# ═════════════════════════════════════════════
# SESSION STATE BOOTSTRAP
# ═════════════════════════════════════════════

def init_session():
    defaults = {
        "all_questions"   : [],      # list[dict] – parsed MCQ records
        "notes_text"      : "",      # concatenated notes text
        "faiss_index"     : None,    # FAISS index object
        "search_results"  : [],      # current search results
        "selected_q"      : None,    # currently displayed question dict
        "last_query"      : "",      # debounce helper
        "pdf_count"       : 0,
        "mcq_file_count"  : 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ═════════════════════════════════════════════
# SIDEBAR  –  Upload & Stats
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 MCQ Smart Search")
    st.markdown("Upload your study material and MCQ files, then search instantly with AI.")
    st.divider()

    # ── PDF Notes Upload ──────────────────────
    st.markdown("### 📄 Upload Notes (PDF)")
    note_files = st.file_uploader(
        "Drop your PDF study notes here",
        type=["pdf"],
        accept_multiple_files=True,
        key="note_uploader",
        help="You can upload multiple PDFs. Text is extracted and stored for context.",
    )

    # ── MCQ Files Upload ─────────────────────
    st.markdown("### 📝 Upload MCQ Questions (PDF / TXT)")
    mcq_files = st.file_uploader(
        "Drop MCQ question files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="mcq_uploader",
        help="Questions are auto-parsed and indexed for semantic search.",
    )

    process_btn = st.button("⚡ Process Files", use_container_width=True, type="primary")

    st.divider()

    # ── Stats ────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stats-box">
            <div class="stats-num">{len(st.session_state.all_questions)}</div>
            <div class="stats-label">Questions</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stats-box">
            <div class="stats-num">{st.session_state.pdf_count + st.session_state.mcq_file_count}</div>
            <div class="stats-label">Files Loaded</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.caption("Powered by · sentence-transformers · FAISS · pdfplumber")


# ═════════════════════════════════════════════
# FILE PROCESSING  (triggered by button)
# ═════════════════════════════════════════════

if process_btn:
    if not note_files and not mcq_files:
        st.warning("⚠️ Please upload at least one file before processing.")
    else:
        with st.spinner("🔍 Reading files and building AI index …"):
            notes_chunks  = []
            all_questions = []

            # ── Extract notes text ───────────────
            for f in (note_files or []):
                text = extract_text_from_pdf(f.read())
                notes_chunks.append(text)

            notes_text = "\n\n".join(notes_chunks)
            st.session_state.notes_text  = notes_text
            st.session_state.pdf_count   = len(note_files or [])

            # ── Extract & parse MCQ files ────────
            for f in (mcq_files or []):
                if f.name.lower().endswith(".pdf"):
                    raw = extract_text_from_pdf(f.read())
                else:
                    raw = extract_text_from_txt(f.read())

                parsed = parse_mcq_blocks(raw)

                # If dedicated parser found nothing, treat each paragraph as a question
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

                all_questions.extend(parsed)

            st.session_state.mcq_file_count = len(mcq_files or [])
            st.session_state.all_questions  = all_questions

            # ── Build FAISS index ────────────────
            if all_questions:
                model = load_embedding_model()
                index, _ = build_faiss_index(all_questions, model)
                st.session_state.faiss_index = index
                st.success(
                    f"✅ Indexed **{len(all_questions)} questions** from "
                    f"**{st.session_state.mcq_file_count}** MCQ file(s) + "
                    f"**{st.session_state.pdf_count}** note PDF(s). Ready to search!"
                )
            else:
                st.warning("No MCQ questions detected. Try a different file format.")


# ═════════════════════════════════════════════
# MAIN PANEL
# ═════════════════════════════════════════════

# ── Header ───────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 MCQ Smart Search</h1>
    <p>AI-powered search across your exam notes and question banks</p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# SEARCH SECTION
# ═════════════════════════════════════════════

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔍 Search Questions</div>', unsafe_allow_html=True)

query = st.text_input(
    label="Type your question here …",
    placeholder="e.g.  what is polymorphism  |  define TCP/IP  |  types of sorting",
    key="search_query",
    label_visibility="collapsed",
)

st.caption("💡 Results update as you type. Click a suggestion to view the full answer.")
st.markdown('</div>', unsafe_allow_html=True)


# ── Perform search on every keystroke ────────
if query and query != st.session_state.last_query:
    st.session_state.last_query = query

    if st.session_state.faiss_index is None or not st.session_state.all_questions:
        st.info("📂 Upload and process your files first (use the sidebar).")
    else:
        model   = load_embedding_model()
        results = semantic_search(
            query,
            model,
            st.session_state.faiss_index,
            st.session_state.all_questions,
            top_k=TOP_K_RESULTS,
        )
        st.session_state.search_results = results
        st.session_state.selected_q     = None   # reset selection on new search


# ═════════════════════════════════════════════
# SUGGESTION PILLS  (top 5 results)
# ═════════════════════════════════════════════

if st.session_state.search_results and query:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">💡 Top {len(st.session_state.search_results)} Matching Questions</div>',
        unsafe_allow_html=True,
    )

    for i, res in enumerate(st.session_state.search_results):
        short_q  = textwrap.shorten(res["question"], width=110, placeholder=" …")
        sim_pct  = int(res["similarity"] * 100)
        btn_label = f"{short_q}  [{sim_pct}% match]"

        if st.button(btn_label, key=f"suggest_{i}", use_container_width=True):
            st.session_state.selected_q = res

    st.markdown('</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════
# ANSWER DISPLAY SECTION
# ═════════════════════════════════════════════

if st.session_state.selected_q:
    q = st.session_state.selected_q

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📖 Question & Answer</div>', unsafe_allow_html=True)

    sim_pct = int(q.get("similarity", 0) * 100)

    # ── Question text ─────────────────────────
    st.markdown(f"""
    <div class="answer-card">
        <div class="question-text">
            ❓ {q['question']}
            <span class="similarity-badge">{sim_pct}% semantic match</span>
        </div>
    """, unsafe_allow_html=True)

    # ── Options ───────────────────────────────
    if q["options"]:
        st.markdown("**Options:**")
        for opt in q["options"]:
            letter = opt[0].upper() if opt else ""
            is_correct = (letter == q.get("correct_key", ""))
            css_class  = "option-correct" if is_correct else "option-wrong"
            badge      = '<span class="correct-badge">✓ CORRECT</span>' if is_correct else ""
            st.markdown(
                f'<div class="option-row {css_class}">{opt}{badge}</div>',
                unsafe_allow_html=True,
            )
    elif q.get("correct_key"):
        st.markdown(f"**✅ Correct Answer: {q['correct_key']}**")

    # ── No options found ──────────────────────
    if not q["options"] and not q.get("correct_key"):
        st.info("ℹ️ No multiple-choice options detected for this entry.")

    # ── Explanation from MCQ file ─────────────
    if q.get("explanation"):
        st.markdown(f"""
        <div class="explanation-box">
            📌 <strong>Explanation:</strong><br>{q['explanation']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)   # close answer-card

    # ── Context from Notes ────────────────────
    if st.session_state.notes_text:
        with st.expander("📚 Related context from your notes", expanded=False):
            snippet = find_context_in_notes(q["question"], st.session_state.notes_text)
            if snippet:
                st.markdown(f"> {snippet}")
            else:
                st.write("No closely matching passage found in uploaded notes.")

    st.markdown('</div>', unsafe_allow_html=True)   # close section-card


# ═════════════════════════════════════════════
# EMPTY STATE  –  shown before any search
# ═════════════════════════════════════════════

elif not query and not st.session_state.all_questions:
    col1, col2, col3 = st.columns(3)
    steps = [
        ("1️⃣", "Upload PDFs", "Add your study notes and MCQ question files using the sidebar."),
        ("2️⃣", "Process Files", "Click ⚡ Process Files to extract text and build the AI index."),
        ("3️⃣", "Search & Learn", "Type any question fragment – AI suggestions appear instantly."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], steps):
        with col:
            st.markdown(f"""
            <div class="section-card" style="text-align:center; min-height:160px;">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:600; color:#1e3a5f; margin:0.4rem 0">{title}</div>
                <div style="font-size:0.88rem; color:#64748b">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

elif not query and st.session_state.all_questions:
    st.success(
        f"✅ **{len(st.session_state.all_questions)} questions** loaded and ready. "
        "Start typing in the search box above!"
    )


# ═════════════════════════════════════════════
# QUESTION BROWSER (full table)
# ═════════════════════════════════════════════

if st.session_state.all_questions:
    with st.expander(
        f"📋 Browse All {len(st.session_state.all_questions)} Questions",
        expanded=False,
    ):
        df = pd.DataFrame([
            {
                "#"          : i + 1,
                "Question"   : textwrap.shorten(q["question"], 120, placeholder="…"),
                "Options"    : len(q["options"]),
                "Answer"     : q.get("correct_key") or "—",
                "Explanation": "✅" if q.get("explanation") else "—",
            }
            for i, q in enumerate(st.session_state.all_questions)
        ])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "#"         : st.column_config.NumberColumn(width="small"),
                "Options"   : st.column_config.NumberColumn("# Options", width="small"),
                "Answer"    : st.column_config.TextColumn(width="small"),
                "Explanation": st.column_config.TextColumn(width="small"),
            },
        )
