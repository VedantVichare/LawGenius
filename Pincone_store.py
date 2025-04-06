import pinecone
import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEndpoint
import fitz  
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
load_dotenv()


def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"  
    return text

pdf_text = extract_text_from_pdf("D:\\ilovepdf_merged.pdf")
doc = fitz.open("D:\\ilovepdf_merged.pdf") 

model = SentenceTransformer("all-MiniLM-L6-v2")  
sentences = pdf_text.split("\n")  
vectors = model.encode(sentences)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone (api_key=PINECONE_API_KEY)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
index=pc.Index(PINECONE_INDEX_NAME)


for page_num in range(0,len(doc)):
    page_text = doc[page_num].get_text("text")  

    if not page_text.strip():  
        continue

    
    page_vector = model.encode(page_text)

    
    if isinstance(page_vector, np.ndarray):
        page_vector = page_vector.tolist()

    
    index.upsert(
        [
            (f"doc-{page_num+1}", page_vector, {"text": page_text})
        ],
        namespace="ns1"
    )

print("PDF stored in vector database successfully!")
