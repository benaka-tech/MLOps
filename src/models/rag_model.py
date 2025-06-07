# RAG (Retrieval-Augmented Generation) Model Integration
# This script demonstrates a simple RAG pipeline using FAISS for vector search and OpenAI's GPT-3.5/4 for generation.
# You can adapt this to your own LLM or vector DB as needed.

import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
# --- Gemini (Google Generative AI) integration ---
import google.generativeai as genai

load_dotenv()

# Set your OpenAI API key (or use another LLM provider)
# openai.api_key = os.getenv('OPENAI_API_KEY')

# Example corpus (replace with your own knowledge base)
corpus = [
    "Customer churn is when a customer stops using a service.",
    "Retention strategies can reduce churn.",
    "Machine learning models can predict churn based on customer data.",
    "Personalized offers can help retain customers.",
    "Subscription-based services often track churn rates."
]

# Step 1: Embed the corpus
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus_embeddings = model.encode(corpus, convert_to_numpy=True)

# Step 2: Build FAISS index
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# Step 3: RAG pipeline function
def rag_query(query, top_k=2):
    # Embed the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    # Retrieve top-k similar documents
    D, I = index.search(query_embedding, top_k)
    retrieved_docs = [corpus[i] for i in I[0]]
    # Compose context
    context = "\n".join(retrieved_docs)
    # Generate answer using Gemini
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = gemini.generate_content(prompt)
    return response.text.strip() if hasattr(response, 'text') else str(response)

# --- RAG Chatbot CLI for Customer Churn Use Case ---
def rag_chatbot():
    print("RAG Chatbot for Customer Churn. Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ('exit', 'quit'): break
        answer = rag_query(query)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    rag_chatbot()
