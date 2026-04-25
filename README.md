# 🤖 RAG Chatbot — AI Document Intelligence System

> Upload PDFs and have intelligent, grounded conversations with your documents.
> Powered by LangChain, Groq (Llama 3.1), FAISS, and HuggingFace Embeddings.

---

## 🚀 Live Demo
> _Deploy link goes here after deployment_

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Multi-PDF Upload | Upload and chat across multiple documents simultaneously |
| 🧠 Smart Onboarding | Auto-generates document summaries and suggested questions on upload |
| 💬 Streaming Responses | Answers stream word-by-word like ChatGPT |
| 🎤 Voice Input | Record or upload audio — transcribed by Groq Whisper |
| 🖼️ Image Analysis | Upload images and ask questions — powered by Llama 4 Vision |
| 📎 Source Citations | Every answer cites the exact file and page number |
| 🧵 Conversation Memory | Full multi-turn memory across the session |
| 📥 PDF Export | Download the entire conversation as a formatted PDF report |

---

## 🏗️ Architecture

```
INGESTION PIPELINE (run once per document)
──────────────────────────────────────────
PDF/DOCX
   ↓
Text Extraction      (PyMuPDF)
   ↓
Chunking             (800 chars, 100 overlap)
   ↓
Embeddings           (all-MiniLM-L6-v2, local)
   ↓
FAISS Vector Store   (saved to disk)

QUERY PIPELINE (every user message)
─────────────────────────────────────────────
User Input (Text / Voice / Image)
   ↓
Embed Question       (same embedding model)
   ↓
Similarity Search    (top-k=4 chunks from FAISS)
   ↓
Build Prompt         (context + question + history)
   ↓
Groq LLM             (Llama 3.1 streaming)
   ↓
Streamed Answer + Source Citations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — Llama 3.1 8B Instant (free) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local, free) |
| Vector Store | FAISS (local) |
| Voice | Groq Whisper large-v3-turbo |
| Vision | Meta Llama 4 Scout (via Groq) |
| Orchestration | LangChain LCEL |
| PDF Parsing | PyMuPDF |
| Frontend | Streamlit |
| PDF Export | fpdf2 |

---

## 📁 Project Structure

```
rag-chatbot/
├── src/
│   ├── app.py          # Streamlit UI (entry point)
│   ├── ingest.py       # PDF parsing + FAISS ingestion
│   ├── llm.py          # LLM, embeddings, RAG chain, streaming
│   ├── audio.py        # Voice transcription (Whisper)
│   ├── vision.py       # Image analysis (Llama Vision)
│   ├── export.py       # PDF report generation
│   └── config.py       # Prompts, model names, settings
├── data/               # Put your PDFs here
├── faiss_index/        # Auto-generated vector store
├── .env                # API keys (never commit!)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
streamlit run src/app.py
```

---

## 📦 Requirements

```txt
langchain
langchain-community
langchain-groq
langchain-huggingface
langchain-core
langchain-text-splitters
faiss-cpu
pymupdf
streamlit
python-dotenv
sentence-transformers
groq
fpdf2
```

Generate with:
```bash
pip freeze > requirements.txt
```

---

## 🐳 Docker

```bash
# Build
docker build -t rag-chatbot .

# Run
docker run -p 8501:8501 --env-file .env rag-chatbot
```

---

## 💡 How It Works

### RAG (Retrieval-Augmented Generation)

Instead of asking a generic LLM that might hallucinate, this system:

1. **Ingests** your PDF — extracts text, splits into chunks, converts to vectors
2. **Stores** those vectors in FAISS (a local vector database)
3. **At query time** — embeds your question, finds the 4 most similar chunks
4. **Builds a prompt** with those chunks as context
5. **Generates** a grounded answer using Llama 3.1 — with citations

The LLM never sees your full document — only the relevant chunks. This makes it scalable and prevents hallucination.

### Why FAISS?
- Runs entirely locally — no cloud costs
- Instant similarity search across millions of vectors
- Persistent — save to disk and reload without re-processing

### Why Groq?
- Free tier available
- Extremely fast inference (fastest Llama hosting available)
- Supports Whisper for voice and Llama Vision for images

---

## 🎯 Interview Q&A

**Q: What is RAG?**
Retrieval-Augmented Generation — retrieving relevant context from a knowledge base before generating an answer. Prevents hallucination and makes responses traceable.

**Q: Why not fine-tune instead?**
Fine-tuning is expensive, slow to update, and doesn't cite sources. RAG is cheaper, updatable in real-time, and every answer can be traced to a source document.

**Q: What is chunking and why does it matter?**
Splitting documents into smaller pieces so each embedding captures focused meaning. Chunk size (800 chars) affects retrieval quality vs context window cost.

**Q: FAISS vs Pinecone?**
FAISS = local, free, in-memory, not persistent across restarts without saving. Pinecone = cloud, managed, persistent, scales to billions of vectors.

**Q: How do you prevent hallucinations?**
Ground answers in retrieved context, set temperature=0, and explicitly instruct the model to say it doesn't know if context is insufficient.

---

## 📝 Resume Bullet Points

```
• Built a multimodal RAG chatbot with LangChain + Groq (Llama 3.1) that ingests
  multi-PDF documents and answers questions grounded in document context with
  source citations, deployed via Streamlit.

• Implemented semantic search using FAISS and HuggingFace sentence-transformers
  for efficient local vector retrieval with no API costs.

• Added multimodal inputs: voice transcription via Groq Whisper and image analysis
  via Llama 4 Vision, with streaming responses and PDF conversation export.

• Architected clean modular codebase (ingest, llm, audio, vision, export modules)
  with smart onboarding — AI-generated document summaries and suggested questions
  on upload.
```

---

## 🔮 What's Next

- [ ] Deploy to Render.com (free tier)
- [ ] Add Pinecone for persistent cloud vector storage
- [ ] Hybrid search (BM25 + vector)
- [ ] RAGAs evaluation metrics
- [ ] FastAPI backend for API access

---

## 📄 License

MIT License — feel free to use and modify.

---

## 🙌 Built With

- [LangChain](https://langchain.com)
- [Groq](https://groq.com)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io)
- [HuggingFace](https://huggingface.co)
