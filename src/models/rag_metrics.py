import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="RAG Metrics & Monitoring", page_icon="📊")
st.title("📊 RAG Model Metrics & Monitoring")

# Example: Show number of documents in the knowledge base
try:
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
