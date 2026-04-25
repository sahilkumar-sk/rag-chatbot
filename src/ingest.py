import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_pdf(pdf_path: str, embeddings, index_path: str = "faiss_index") -> FAISS:
    """Ingest a single PDF — creates or MERGES into existing FAISS index."""
    pdf_file = Path(pdf_path)
    index_dir = Path(index_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"File not found: {pdf_file}")

    print(f"📄 Loading PDF: {pdf_file.name}")
    loader = PyMuPDFLoader(str(pdf_file))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = pdf_file.name

    print(f"✅ Loaded {len(documents)} pages")

    print("✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    print("💾 Storing in FAISS...")
    new_vectorstore = FAISS.from_documents(chunks, embeddings)

    if index_dir.exists():
        print("🔀 Merging with existing index...")
        existing = FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True
        )
        existing.merge_from(new_vectorstore)
        existing.save_local(str(index_dir))
        print(f"✅ Merged!")
        return existing
    else:
        index_dir.mkdir(parents=True, exist_ok=True)
        new_vectorstore.save_local(str(index_dir))
        print(f"✅ Saved!")
        return new_vectorstore


def ingest_multiple_pdfs(pdf_paths: list, embeddings, index_path: str = "faiss_index") -> FAISS:
    """Ingest multiple PDFs into one combined FAISS index."""
    import shutil
    if Path(index_path).exists():
        shutil.rmtree(index_path)
        print("🗑️  Cleared old index")

    vectorstore = None
    for path in pdf_paths:
        vectorstore = ingest_pdf(path, embeddings, index_path)

    return vectorstore


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/ingest.py data/file1.pdf data/file2.pdf")
        sys.exit(1)
    ingest_multiple_pdfs(sys.argv[1:])