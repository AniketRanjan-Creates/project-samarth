# 🇮🇳 Project Samarth — Intelligent Agri Data Q&A System

An intelligent, tricolor-themed Q&A system that answers **natural language questions** about **India’s agricultural mandi (market) data** using **Groq’s Llama 3** and **retrieval-augmented generation (RAG)** — powered by **Agmarknet** datasets and built with **Streamlit**.

---

## 🎯 Vision

Government datasets like those on [data.gov.in](https://data.gov.in) are rich but fragmented.  
**Project Samarth** bridges that gap — allowing users to **ask questions in plain English** and get **data-backed answers** about mandi arrivals, crop varieties, and price trends, cited directly from official sources.

---

## 🌾 Features

- 🧠 **LLM-Powered Q&A:** Uses **Groq Llama 3.1 8B Instant** for intelligent reasoning.  
- 🏬 **Government Data Integration:** Draws from Agmarknet mandi datasets.  
- 🗂️ **Embeddings + RAG:** Uses `HuggingFace all-MiniLM-L6-v2` embeddings.  
- 💾 **In-Memory Chroma Index:** No SQLite files — built dynamically from `documents.json` for Streamlit Cloud compatibility.  
- 🎨 **Beautiful UI:** Custom saffron–white–green theme matching India’s tricolor.  
- 📦 **Data Export:** Download search results as CSV directly from the app.  
- 🪶 **Citations:** Every answer includes official attribution to Agmarknet (GOI).  

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **LLM** | Groq Llama 3.1 (via Groq API) |
| **Vector DB** | Chroma (in-memory) |
| **Embeddings** | HuggingFace Sentence Transformer |
| **Language Chain** | LangChain |
| **Deployment** | Streamlit Cloud |
| **Dataset Source** | Agmarknet (Government of India) |

--------

## 🌐 Live Demo

[https://project-samarth-defkoltsssfjswxwttg3mg.streamlit.app/]

------

🧭 Author

Aniket Ranjan

