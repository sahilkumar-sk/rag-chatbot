import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PROMPT_TEMPLATE = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided document."
Do NOT make up information.

Context:
{context}

Question:
{question}

Answer:
"""


def load_vectorstore(index_path: str = "faiss_index") -> FAISS:
    """Load the previously saved FAISS index from disk."""
    print("📂 Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded!")
    return vectorstore


def format_docs(docs):
    """Combine retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(vectorstore: FAISS):
    """Build the RAG chain using modern LangChain LCEL syntax."""

    # LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Retriever — fetch top 4 relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # LCEL Chain: retriever → prompt → llm → parse output
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain ready!")
    return retriever, chain


# ---------- Quick test ----------
if __name__ == "__main__":
    vectorstore = load_vectorstore()
    retriever, chain = build_chain(vectorstore)

    print("\n🤖 Ask a question about your document (type 'quit' to exit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        # Get answer
        answer = chain.invoke(question)

        # Get source pages separately
        source_docs = retriever.invoke(question)
        sources = set()
        for doc in source_docs:
            page = doc.metadata.get("page", "?")
            sources.add(f"Page {int(page) + 1}")

        print(f"\n💬 Answer:\n{answer}")
        print(f"📎 Sources: {', '.join(sorted(sources))}\n")