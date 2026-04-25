import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import (
    CHAT_SYSTEM_PROMPT, SUMMARY_PROMPT, QUESTIONS_PROMPT,
    LLM_MODEL, EMBEDDING_MODEL, RETRIEVER_K
)

load_dotenv()


@st.cache_resource
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm(streaming: bool = False) -> ChatGroq:
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=streaming
    )


def get_groq_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def format_docs(docs) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        parts.append(f"[From: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_chain(vectorstore: FAISS):
    """Build retriever and RAG chain."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    llm = get_llm()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chain


def generate_doc_summary(filename: str, vectorstore: FAISS) -> str:
    """Generate a 2-3 sentence summary for a single document."""
    llm = get_llm()
    all_docs = vectorstore.similarity_search(filename, k=5)
    doc_chunks = [d for d in all_docs if d.metadata.get("source_file") == filename]
    if not doc_chunks:
        doc_chunks = all_docs[:3]
    content = " ".join(d.page_content for d in doc_chunks[:3])[:1500]
    prompt = SUMMARY_PROMPT.format(filename=filename, content=content)
    return llm.invoke(prompt).content


def generate_suggested_questions(summaries: dict) -> list:
    """Generate 5 suggested questions across all documents."""
    llm = get_llm()
    combined = "\n".join(f"- {fname}: {s}" for fname, s in summaries.items())
    prompt = QUESTIONS_PROMPT.format(summary=combined)
    response = llm.invoke(prompt)
    lines = response.content.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit():
            q = line.split(".", 1)[-1].strip()
            if q:
                questions.append(q)
    return questions[:5]


def stream_question(question: str, msg_type: str = "text"):
    """Stream a RAG answer word by word."""
    import streamlit as st

    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "type": msg_type
    })

    source_docs = st.session_state.retriever.invoke(question)
    sources = sorted(set(
        f"{doc.metadata.get('source_file', '?')} p.{int(doc.metadata.get('page', 0)) + 1}"
        for doc in source_docs
    ))
    sources_str = "Sources: " + ", ".join(sources)
    context = format_docs(source_docs)

    messages = [SystemMessage(content=CHAT_SYSTEM_PROMPT.format(context=context))]
    messages.extend(st.session_state.chat_history)
    messages.append(HumanMessage(content=question))

    llm = get_llm(streaming=True)
    full_answer = ""

    with st.chat_message("assistant"):
        full_answer = st.write_stream(
            chunk.content
            for chunk in llm.stream(messages)
            if chunk.content
        )
        st.caption(f"📎 {sources_str}")

    st.session_state.chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=full_answer)
    ])
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources_str,
        "type": "text"
    })