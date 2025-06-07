import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.rag_model import rag_query

st.set_page_config(page_title="Customer Churn RAG Portal", page_icon="💬")
st.title("💬 Customer Churn RAG Portal")

selected = option_menu(
    menu_title=None,
    options=["Chatbot", "Metrics & Monitoring"],
    icons=["chat-dots", "bar-chart"],
    orientation="horizontal"
)

if selected == "Chatbot":
    st.header("RAG Chatbot")
    st.write("Ask any question about customer churn, retention, or your data!")
    st.markdown("**Sample questions:**\n- Why do customers churn?\n- How can I reduce churn in my subscription service?\n- What features are most important for predicting churn?\n- Give me a summary of churn rates in the data.\n- Suggest retention strategies for high-risk customers.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    user_input = st.text_input("You:", "", key="user_input")
    if st.button("Send") and user_input.strip():
        answer = rag_query(user_input)
        st.session_state.chat_history.append((user_input, answer))
        st.session_state.last_user_input = user_input
    if st.button("New Chat"):
        st.session_state.chat_history = []
        st.session_state.last_user_input = ""
    for user, bot in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {user}")
        st.markdown(f"**Bot:** {bot}")

if selected == "Metrics & Monitoring":
    st.header("RAG Model Metrics & Monitoring")
    try:
        import pandas as pd
        kb_path = os.path.join(os.path.dirname(__file__), '../../data/processed_data.csv')
        kb_df = pd.read_csv(kb_path)
        st.metric("Knowledge Base Size (rows)", len(kb_df))
    except Exception as e:
        st.warning(f"Could not load knowledge base: {e}")
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
    # MLOps Metrics Section
    st.subheader("MLOps Metrics")
    # Example: Model file size
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../../data/churn_model.pkl')
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / 1024  # KB
            st.metric("Model File Size (KB)", f"{model_size:.2f}")
    except Exception as e:
        st.warning(f"Could not get model file size: {e}")
    # Example: Model last trained timestamp
    try:
        import datetime
        if os.path.exists(model_path):
            last_trained = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
            st.metric("Model Last Trained", last_trained.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        st.warning(f"Could not get model last trained time: {e}")
    # Example: Model accuracy, precision, recall, f1, ROC curve, confusion matrix (if available)
    try:
        metrics_path = os.path.join(os.path.dirname(__file__), '../../data/model_metrics.csv')
        if os.path.exists(metrics_path):
            import pandas as pd
            import matplotlib.pyplot as plt
            metrics_df = pd.read_csv(metrics_path)
            if 'accuracy' in metrics_df.columns:
                st.metric("Model Accuracy", f"{metrics_df['accuracy'].iloc[-1]:.2%}")
            if 'precision' in metrics_df.columns:
                st.metric("Precision", f"{metrics_df['precision'].iloc[-1]:.2%}")
            if 'recall' in metrics_df.columns:
                st.metric("Recall", f"{metrics_df['recall'].iloc[-1]:.2%}")
            if 'f1' in metrics_df.columns:
                st.metric("F1 Score", f"{metrics_df['f1'].iloc[-1]:.2%}")
            # Line chart for accuracy over time
            if 'accuracy' in metrics_df.columns and 'timestamp' in metrics_df.columns:
                st.line_chart(metrics_df.set_index('timestamp')['accuracy'], use_container_width=True)
            # ROC curve
            if 'fpr' in metrics_df.columns and 'tpr' in metrics_df.columns:
                fig, ax = plt.subplots()
                ax.plot(metrics_df['fpr'], metrics_df['tpr'], label='ROC Curve')
                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend()
                st.pyplot(fig)
            # Confusion matrix
            if 'cm_00' in metrics_df.columns:
                import numpy as np
                import seaborn as sns
                cm = np.array([
                    [metrics_df['cm_00'].iloc[-1], metrics_df['cm_01'].iloc[-1]],
                    [metrics_df['cm_10'].iloc[-1], metrics_df['cm_11'].iloc[-1]]
                ])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not load or plot model metrics: {e}")
