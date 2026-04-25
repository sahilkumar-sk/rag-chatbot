# All prompts and settings in one place

CHAT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about uploaded documents.
Use the context below to answer. If the answer isn't in the context, say you don't know.
Be conversational and remember what was discussed earlier.
When relevant, mention which document the information came from.

Context from documents:
{context}"""

SUMMARY_PROMPT = """Summarize this document excerpt in 2-3 sentences.
Be concise and highlight the core topic and purpose.

Document: {filename}
Content: {content}

Summary:"""

QUESTIONS_PROMPT = """Based on these documents, generate exactly 5 insightful questions
a user would genuinely want to ask. Make them specific and interesting, not generic.
Return ONLY a numbered list 1-5, no extra text.

Documents summary: {summary}

Questions:"""

# Model settings
LLM_MODEL = "llama-3.1-8b-instant"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
WHISPER_MODEL = "whisper-large-v3-turbo"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_K = 4