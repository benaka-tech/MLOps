import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.rag_model import rag_query

st.set_page_config(page_title="Customer Churn RAG Chatbot", page_icon="💬")
st.title("💬 Customer Churn RAG Chatbot")
st.write("Ask any question about customer churn, retention, or your data!")

st.subheader("Use Case: Customer Churn Prediction & RAG Chatbot")
st.markdown("""
This chatbot assists with:
- Answering questions about customer churn and retention strategies
- Providing insights from your data and knowledge base
- Supporting customer service and business analysis

**How it works:**
- Your question is matched to relevant knowledge using vector search (RAG)
- An LLM generates a helpful answer using the retrieved context

_Example questions:_
- Why do customers churn?
- How can I reduce churn in my subscription service?
- What features are most important for predicting churn?
""")

st.subheader("RAG Model Metrics & Monitoring")

# Example: Show number of documents in the knowledge base
try:
    import pandas as pd
    import os
    kb_path = os.path.join(os.path.dirname(__file__), '../../data/processed_data.csv')
    kb_df = pd.read_csv(kb_path)
    st.metric("Knowledge Base Size (rows)", len(kb_df))
except Exception as e:
    st.warning(f"Could not load knowledge base: {e}")

# Example: Show FAISS index stats (if available)
try:
    import faiss
    import numpy as np
    emb_path = os.path.join(os.path.dirname(__file__), '../../data/rag_embeddings.npy')
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        st.metric("Vector Index Size", emb.shape[0])
        st.metric("Embedding Dimension", emb.shape[1])
except Exception as e:
    st.warning(f"Could not load vector index: {e}")

# Example: Chatbot usage monitoring (session-based)
if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0

user_input = st.text_input("You:", "", key="user_input")

if st.session_state.get("last_user_input") != user_input and user_input.strip():
    st.session_state.chat_count += 1
    st.session_state.last_user_input = user_input
st.metric("Chatbot Interactions (this session)", st.session_state.chat_count)

# Example: Graph - Churn Distribution from KB
try:
    if 'churn' in kb_df.columns:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots()
        sns.countplot(x='churn', data=kb_df, ax=ax)
        ax.set_title('Churn Distribution in Knowledge Base')
        st.pyplot(fig)
except Exception as e:
    st.warning(f"Could not plot churn distribution: {e}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add a New Chat button to clear chat history and reset input
if st.button("New Chat"):
    st.session_state.chat_history = []
    st.session_state.last_user_input = ""
    # Optionally, clear the input box if possible

if st.button("Send") and user_input.strip():
    answer = rag_query(user_input)
    st.session_state.chat_history.append((user_input, answer))
    st.session_state.last_user_input = user_input
    # Do not attempt to clear st.session_state.user_input directly
    # Instead, suggest to the user to clear the input manually or use a workaround if needed

for user, bot in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {user}")
    st.markdown(f"**Bot:** {bot}")
