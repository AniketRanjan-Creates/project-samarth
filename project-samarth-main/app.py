# app.py — Project Samarth (Final UI with saffron section titles)
import sys
import os
import json
import logging
from datetime import datetime

import pandas as pd
import streamlit as st
import chromadb

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import *  # CHROMA_DB_PATH, COLLECTION_NAME

# ====== Env & Logging ======
load_dotenv()
RAW_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

def is_valid_groq_key(k: str) -> bool:
    return isinstance(k, str) and k.startswith("gsk_") and len(k) > 8

GROQ_KEY = RAW_GROQ_KEY if is_valid_groq_key(RAW_GROQ_KEY) else ""

sys.setrecursionlimit(3000)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== Page Setup ======
st.set_page_config(page_title="Project Samarth", page_icon="🇮🇳", layout="wide")

# ====== Tricolour Theme (Saffron–White–Green) ======
st.markdown(
    """
    <style>
      :root{
        --saffron:#FF9933;
        --green:#138808;
        --bg:#ffffff;
        --ink:#0f172a;
        --muted:#475569;
        --border:#e2e8f0;
      }
      html, body, [class*="css"] { background: var(--bg) !important; }
      .hero {
        border-radius:16px;
        padding:18px 22px;
        border:1px solid var(--border);
        background: linear-gradient(90deg, var(--saffron) 0%, #ffffff 50%, var(--green) 100%);
        box-shadow:0 2px 8px rgba(0,0,0,0.06);
        margin-bottom:14px;
        text-align:center;
      }
      .hero h1{
        margin:0;
        font-size:28px;
        color:#111827;
        letter-spacing:.2px;
        text-shadow: 0 1px 0 rgba(255,255,255,0.6);
      }
      .sub{
        margin:6px 0 0 0;
        color:#111827;
        font-size:14px;
        font-weight:500;
      }
      .badge {
        display:inline-block; margin-left:8px; padding:3px 8px; border-radius:999px;
        font-size:12px; font-weight:700; color:#fff;
        background: rgba(0,0,0,0.35);
      }

      /* 🟠 Saffron section titles (Your Question / Examples / Preview) */
      .section-title{
        font-weight:800; 
        color:#ffffff; /* bright white text */
        font-size:18px; 
        margin:12px 0 8px 0;
        border-left:5px solid var(--saffron); 
        padding:10px 12px;
        background: linear-gradient(90deg, var(--saffron) 0%, rgba(255,153,51,0.85) 100%);
        border-radius:8px;
        box-shadow:0 2px 4px rgba(0,0,0,0.08);
      }

      .stButton>button{
        background:var(--green) !important; color:#fff !important; border:none; border-radius:8px !important;
      }
      .stButton>button:hover{ background:#0e6e06 !important; }
      .footer{
        color:var(--muted); font-size:12px; text-align:center;
      }
      .block-container{ padding-top:10px; }
      .sidebar-flag{
        margin-top:12px;
        text-align:left;
      }
      .sidebar-flag img{
        width:130px; border-radius:6px; box-shadow:0 1px 4px rgba(0,0,0,0.2);
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ====== Header ======
st.markdown(
    f"""
    <div class="hero">
      <h1>Project Samarth <span class="badge">{'LLM Active' if GROQ_KEY else 'LLM Inactive'}</span></h1>
      <p class="sub">Professional Q&A on Indian <b>mandi prices & arrivals</b> (Agmarknet). Clean answers, tables, CSV export, and citations.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if not GROQ_KEY:
    st.warning("No valid GROQ_API_KEY found — Q&A is disabled. Add it in `.env` to enable answers.")

# ====== Core: VectorStore & optional LLM ======
@st.cache_resource(ttl=3600)
def initialize_qa_system():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _ = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(client=chroma_client, collection_name=COLLECTION_NAME, embedding_function=embeddings)

        llm = None
        if GROQ_KEY:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_KEY, temperature=0.2)
            logger.info("Groq LLM initialized.")

        prompt_template = """You are an assistant for India's agricultural mandi data.
Data fields: state, district, market, commodity, variety, grade, arrival_date, min_price, max_price, modal_price.
Answer only using these fields. Always cite: Agmarknet (Government of India) market arrivals and prices.

Context:
{context}

Question:
{question}

Answer:
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        qa_chain = None
        if llm:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT},
            )
        return qa_chain, vectorstore
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        return None, None

def build_rows_from_sources(result_dict, limit=30):
    rows, seen = [], set()
    for d in (result_dict or {}).get("source_documents", [])[:limit]:
        m = d.metadata or {}
        key = (m.get("state",""), m.get("district",""), m.get("market",""),
               m.get("commodity",""), m.get("arrival_date",""))
        if key in seen: continue
        seen.add(key)
        rows.append({
            "State": m.get("state",""),
            "District": m.get("district",""),
            "Market": m.get("market",""),
            "Commodity": m.get("commodity",""),
            "Variety": m.get("variety",""),
            "Grade": m.get("grade",""),
            "Date": m.get("arrival_date", m.get("date","")),
            "Min (₹/qtl)": m.get("min_price",""),
            "Max (₹/qtl)": m.get("max_price",""),
            "Modal (₹/qtl)": m.get("modal_price",""),
        })
    return rows

# ====== Sidebar ======
with st.sidebar:
    st.markdown("### 📚 Data Sources")
    st.write("• **Agmarknet (GOI)** — arrivals & prices")
    st.caption("Directorate of Marketing & Inspection, MoA&FW")

    st.markdown("### ⚙️ System Status")
    st.write(f"LLM Key: {'✅ Active' if GROQ_KEY else '⚠️ Missing'}")
    st.write("Vector DB: ✅ Chroma (local, persistent)")
    st.divider()
    st.caption("All processing is local. `.env` stores keys. CSV export enabled. Citations attached.")

    # Bottom-left flag
    st.markdown(
        """
        <div class="sidebar-flag">
            <img src="https://upload.wikimedia.org/wikipedia/commons/7/7b/India_flag_300.png" alt="India Flag">
        </div>
        """,
        unsafe_allow_html=True
    )

# ====== Tabs ======
tab_qa, tab_preview = st.tabs(["💬 Ask a Question", "🌦️ Climate × Agri (Preview)"])

with tab_qa:
    st.markdown('<div class="section-title">Examples</div>', unsafe_allow_html=True)
    examples = [
        "Modal price of tomato in Chittoor on 25/10/2025?",
        "Highest price for banana in Mehsana on 25/10/2025?",
        "Commodities traded in Guntur market on 25/10/2025.",
        "Varieties of cabbage traded in Amreli?",
        "Price range for dry chillies in Guntur on 25/10/2025?",
        "What is the price of brinjal in Bilimora today?",
    ]
    ex_cols = st.columns(3)
    for i, q in enumerate(examples):
        with ex_cols[i % 3]:
            if st.button(q, key=f"ex_{i}"):
                st.session_state["current_question"] = q

    st.markdown('<div class="section-title">Your Question</div>', unsafe_allow_html=True)
    user_q = st.text_area(
        label="",
        value=st.session_state.get("current_question", ""),
        height=90,
        placeholder="e.g., Modal price of tomato in Chittoor on 25/10/2025?",
    )

    go = st.button("🔎 Get Answer", disabled=(not GROQ_KEY))
    if go and user_q.strip():
        with st.spinner("Reasoning over indexed mandi data…"):
            qa_chain, _ = initialize_qa_system()
            if qa_chain is None:
                st.warning("LLM is disabled or failed to initialize. Add a valid GROQ_API_KEY in `.env`.")
            else:
                result = qa_chain.invoke({"query": user_q})
                st.markdown("#### Answer")
                st.write(result["result"])

                st.markdown("#### Details (top matches)")
                rows = build_rows_from_sources(result, limit=30)
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download CSV", csv, "mandi_results.csv", "text/csv")
                else:
                    st.caption("No structured rows extracted from sources.")

                st.caption("Source: Agmarknet (Government of India) — market arrivals and prices")

with tab_preview:
    st.markdown('<div class="section-title">Climate × Agriculture (Preview)</div>', unsafe_allow_html=True)
    st.info("Future feature: integrate IMD rainfall & DES crop production datasets for trend and correlation analysis.")

# ====== Footer ======
st.markdown("---")
st.markdown(
    f'<div class="footer">Last updated: {datetime.now().strftime("%d %b %Y, %I:%M %p")} • '
    'Accuracy & Traceability • Privacy-first (local processing)</div>',
    unsafe_allow_html=True
)
