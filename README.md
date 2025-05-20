# 🧠 LawGenius – AI-Powered Legal Chatbot

[![Streamlit App](https://img.shields.io/badge/Try%20it%20Live-Streamlit-green)](https://lawgenius-nf6kr3wsdewsoftwotqruj.streamlit.app/)

LawGenius is an AI-powered legal assistant that combines **Generative AI (GenAI)**, **Retrieval-Augmented Generation (RAG)**, and **Legal-BERT** to answer legal queries with context-aware, human-like responses. It supports document-based querying as well, allowing users to upload legal documents and ask questions specifically from that content.

---

## 📌 Features

- ✅ Ask general legal questions and get human-like answers.
- ✅ Upload your own legal documents and ask questions based on them.
- ✅ Combines Gemini API with Legal-BERT and a RAG pipeline.
- ✅ Uses Pinecone for semantic vector search.
- ✅ Built and deployed using Streamlit.

---

## 🧠 Tech Stack

- **Gemini API** (Gemini or Gemini Pro)
- **Legal-BERT** (for legal language understanding)
- **RAG (Retrieval-Augmented Generation)** concept
- **Pinecone** (vector DB for storing book embeddings)
- **sentence-transformers** (for embedding generation)
- **Streamlit** (for the web interface)
- **Python**

---

## 🔁 Workflow

### 📘 General Legal QA

1. User inputs a legal question.
2. Question is embedded using `sentence-transformers`.
3. Embedding is queried in Pinecone for matching book content.
4. Matched content is passed through **Legal-BERT**.
5. Result + context is sent to **GenAI (Gemini)**.
6. Gemini produces a human-like response.

### 📄 Document-Based QA

1. User uploads a legal document (PDF or text).
2. When a question is asked, it is converted into an embedding.
3. The question embedding is used to query Pinecone and retrieve relevant document chunks.<br>
4. The Legal-BERT also find answer.<br>
5. Then both pincone and Legal-BERT data given to the gemini API to generate human-like response.<br>

---

## 🚀 Deployment

The app is deployed using Streamlit.  
👉 [Click here to try LawGenius](https://lawgenius-nf6kr3wsdewsoftwotqruj.streamlit.app/)

---


