import os
import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Retrieve API keys from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

st.set_page_config(page_title="📜 Legal AI Chatbot", layout="wide")
st.title("📜 Legal AI Chatbot")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Load embedding model
MODEL_PATH = "./model_dir"
embed_model = SentenceTransformer(MODEL_PATH)

# Load LegalBERT model
LEGAL_BERT_PATH = "./legalbert_model1"
tokenizer = AutoTokenizer.from_pretrained(LEGAL_BERT_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(LEGAL_BERT_PATH)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

def extract_legal_document(pdf_file):
    text = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text) if text else "No extractable text found in the document."
    except Exception as e:
        return f"Error extracting text: {e}"

# Session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "legal_document_text" not in st.session_state:
    st.session_state.legal_document_text = ""

# Sidebar: PDF Upload
with st.sidebar:
    uploaded_file = st.file_uploader("📂 Upload a Legal Document (PDF)", type=["pdf"])
    if uploaded_file:
        extracted_text = extract_legal_document(uploaded_file)
        if "Error" in extracted_text:
            st.error(extracted_text)
        else:
            st.session_state.legal_document_text = extracted_text
            st.success("✅ Document uploaded successfully!")

# Chat UI
st.subheader("💬 Chat with Legal AI")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a legal question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        # Embed the question
        q_vector = embed_model.encode(question).tolist()

        # Search Pinecone index
        matches = index.query(
            namespace="legal_ns",
            vector=q_vector,
            top_k=3,
            include_metadata=True
        ).get("matches", [])

        # Gather retrieved texts
        retrieved_context = "\n".join([match["metadata"].get("text", "No relevant information found.") for match in matches])

        # QA with LegalBERT
        legal_qa_response = ""
        if st.session_state.legal_document_text:
            legal_qa_result = qa_pipeline({
                "question": question,
                "context": st.session_state.legal_document_text[:2000]
            })
            legal_qa_response = legal_qa_result["answer"]

        # Final Gemini Prompt
        final_prompt = f"""
        Act as a professional legal expert. Answer legal queries with precision and clarity.
        If a legal document is provided, use its content for reference. Otherwise, provide general legal advice.

        ### User Question: {question}
        ### Context from Legal Documents: {st.session_state.legal_document_text[:1000]}
        ### Retrieved Legal References: {retrieved_context}
        ### Legal-BERT Answer: {legal_qa_response}

        Provide a structured and well-explained legal response.
        """

        gemini_response = gemini_model.generate_content(final_prompt).text

        # Display response
        st.session_state.messages.append({"role": "assistant", "content": gemini_response})
        with st.chat_message("assistant"):
            st.markdown(gemini_response)

    except Exception as e:
        st.error(f"Error: {e}")
