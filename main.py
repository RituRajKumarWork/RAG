import streamlit as st
from qa_engine.document_loader import load_documents, chunk_documents
from qa_engine.embedder import create_vector_store
from qa_engine.qa_chain import build_qa_chain

# Setup page
st.set_page_config(page_title="LangChain DocBot 💬", layout="wide")
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("<div class='header'><h1>🧠 LangChain DocBot</h1><p>Chat with your documents – with local LLM magic ✨</p></div>", unsafe_allow_html=True)

# File upload
uploaded_files = st.file_uploader("📄 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

# Session state init
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Load, embed & build chain
if uploaded_files:
    with st.spinner("Reading and embedding documents..."):
        docs = load_documents(uploaded_files)
        chunks = chunk_documents(docs)
        vector_store = create_vector_store(chunks)
    st.session_state.chain = build_qa_chain(vector_store)
    st.success("✅ Files uploaded successfully! Now ask your questions.")

# --- Chat history always on top ---
if st.session_state.chain:
    # Display chat top-down
    for chat in st.session_state.chat_history:
        st.markdown(f"""
        <div class='chat-bubble user'><span>👤</span><p>{chat['question']}</p></div>
        <div class='chat-bubble bot'><span>🤖</span><p>{chat['answer']}</p></div>
        """, unsafe_allow_html=True)

    # Sticky input bar at bottom
    st.markdown("<div class='chat-title'>💬 Ask a Question</div>", unsafe_allow_html=True)
    st.session_state.user_input = st.text_area(
        "Type your question...",
        value=st.session_state.user_input,
        height=80,
        placeholder="e.g., What is dynamic programming?",
        label_visibility="collapsed"
    )

    if st.button("✈️ Send", use_container_width=True) and st.session_state.user_input.strip():
        question = st.session_state.user_input
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"question": question})
            answer = response["answer"]

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

        st.session_state.user_input = ""
        st.rerun()