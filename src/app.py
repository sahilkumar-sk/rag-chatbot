import os
import sys
import tempfile
import shutil
import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(os.path.dirname(__file__))
from ingest import ingest_multiple_pdfs
from config import LLM_MODEL
from llm import (
    load_embeddings, get_groq_client, build_chain,
    generate_doc_summary, generate_suggested_questions,
    stream_question, format_docs
)
# from audio import transcribe_audio
from vision import analyze_image
from export import generate_chat_pdf

# ═══════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("Upload PDFs - chat with text or images.")

# ═══════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.caption(f"- {f.name}")

    if uploaded_files and st.button("🔄 Process PDFs", use_container_width=True):
        with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
            tmp_paths = []
            for uf in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf",
                    prefix=uf.name.replace(".pdf", "") + "_") as tmp:
                    tmp.write(uf.read())
                    tmp_paths.append((tmp.name, uf.name))

            final_paths = []
            for tmp_path, original_name in tmp_paths:
                final_path = str(Path(tempfile.gettempdir()) / original_name)
                shutil.copy(tmp_path, final_path)
                final_paths.append(final_path)

            embeddings = load_embeddings()
            vectorstore = ingest_multiple_pdfs(final_paths, embeddings)
            retriever, chain = build_chain(vectorstore)

            st.session_state.update({
                "retriever": retriever,
                "chain": chain,
                "messages": [],
                "chat_history": [],
                "doc_names": [f.name for f in uploaded_files],
                "vectorstore": vectorstore
            })

        with st.spinner("🧠 Generating summaries..."):
            summaries = {name: generate_doc_summary(name, vectorstore)
                        for name in st.session_state.doc_names}
            st.session_state.summaries = summaries

        with st.spinner("💡 Generating questions..."):
            st.session_state.suggested_questions = generate_suggested_questions(summaries)

        st.success(f"✅ {len(uploaded_files)} PDF(s) ready!")

    if "doc_names" in st.session_state:
        st.divider()
        st.write("**📚 Loaded Documents:**")
        for name in st.session_state.doc_names:
            st.caption(f"- {name}")

    if st.session_state.get("messages"):
        st.divider()
        if st.button("📥 Export Chat as PDF", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_bytes = generate_chat_pdf(
                    st.session_state.messages,
                    st.session_state.get("doc_names", [])
                )
            st.download_button(
                label="⬇️ Download Report",
                data=bytes(pdf_bytes),
                file_name=f"rag_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    st.caption("Built with LangChain + Groq + FAISS")

# ═══════════════════════════════════════
# INIT STATE
# ═══════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "voice_counter" not in st.session_state:
    st.session_state.voice_counter = 0

# ═══════════════════════════════════════
# HANDLE PENDING QUESTION
# ═══════════════════════════════════════
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    with st.chat_message("user"):
        st.write(question)
    stream_question(question, msg_type="text")
    st.rerun()

# ═══════════════════════════════════════
# CHAT HISTORY DISPLAY
# ═══════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            if msg.get("type") == "image":
                st.caption("🖼️ Image question")
        st.write(msg["content"])
        if "sources" in msg:
            st.caption(f"📎 {msg['sources']}")

# ═══════════════════════════════════════
# INPUT TABS
# ═══════════════════════════════════════
if "chain" not in st.session_state:
    st.info("👈 Upload and process PDFs to start chatting.")
else:
    tab_text, tab_image = st.tabs(["💬 Text", "🖼️ Image"])

    # ── TEXT ──
    with tab_text:
        if question := st.chat_input("Ask anything about your documents..."):
            with st.chat_message("user"):
                st.write(question)
            stream_question(question, msg_type="text")
            st.rerun()

    # ── IMAGE ──
    with tab_image:
        st.write("**Upload an image and ask a question about it.**")
        uploaded_image = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png", "webp"], key="image_upload"
        )
        if uploaded_image:
            st.image(uploaded_image, use_column_width=True)
            image_question = st.text_input(
                "What do you want to know?",
                placeholder="Summarize this chart / What does this show?"
            )
            if st.button("🔍 Analyze Image", use_container_width=True):
                if not image_question:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Analyzing..."):
                        doc_context = format_docs(
                            st.session_state.retriever.invoke(image_question)
                        ) if "retriever" in st.session_state else ""
                        answer = analyze_image(
                            uploaded_image.read(),
                            image_question,
                            get_groq_client(),
                            doc_context
                        )
                    display_q = f"[Image: {uploaded_image.name}] {image_question}"
                    st.session_state.messages.extend([
                        {"role": "user", "content": display_q, "type": "image"},
                        {"role": "assistant", "content": answer, "type": "text"}
                    ])
                    st.session_state.chat_history.extend([
                        HumanMessage(content=display_q),
                        AIMessage(content=answer)
                    ])
                    st.rerun()