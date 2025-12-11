# rag_pipeline.py
import os, json
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

from classifier import predict_system

load_dotenv()

DATA_PATH = "full_systems_dataset_fixed.json"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# =====================
# Load Legal Documents
# =====================
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []
for item in data:
    page_content = f"{item['system']} - {item['article_number']}:\n{item['text']}"
    metadata = {
        "system": item["system"],
        "system_code": item["system_code"],
        "article_key": item["article_key"],
        "article_number": item["article_number"],
        "original_text": item["text"],
    }
    documents.append(Document(page_content=page_content, metadata=metadata))

# =====================
# Retrievers
# =====================
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
faiss_store = FAISS.from_documents(documents, embeddings)
faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 8})

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 8

def hybrid_retrieve(query: str, k=5, system_hint=None):
    bm25_docs = bm25_retriever.invoke(query)
    faiss_docs = faiss_retriever.invoke(query)

    merged, seen = [], set()

    def add(docs):
        for d in docs:
            if system_hint and d.metadata.get("system_code") != system_hint:
                continue
            key = d.metadata.get("article_key")
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
            if len(merged) >= k:
                break

    add(bm25_docs)
    if len(merged) < k:
        add(faiss_docs)

    # fallback آمن
    if not merged:
        add(bm25_docs)
        add(faiss_docs)

    return merged[:k]

# =====================
# Prompt Builder
# =====================
client = OpenAI()

def build_prompt(question, docs):
    context = ""
    for i, d in enumerate(docs, 1):
        m = d.metadata
        context += (
            f"[{i}] {m['system']} ({m['system_code']}) "
            f"{m['article_number']} – {m['article_key']}\n"
            f"{m['original_text']}\n\n"
        )

    return f"""
أنت مساعد قانوني متخصص في الأنظمة السعودية.
اعتمد فقط على النصوص أدناه، ولا تجتهد خارجها.

النصوص النظامية:
{context}

السؤال:
{question}

التعليمات:
- أجب بالعربية الفصحى.
- اربط كل حكم بالمراجع [1] [2].
- إذا لم تكفِ النصوص، صرّح بذلك.
""".strip()

# =====================
# Public API Function
# =====================
def answer_question(question: str):
    system_hint = predict_system(question)
    print(f"[Router] Suggested system: {system_hint}")

    docs = hybrid_retrieve(question, k=5, system_hint=system_hint)

    if not docs:
        return {"answer": "لا توجد نصوص كافية للإجابة.", "articles": []}

    prompt = build_prompt(question, docs)

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "articles": [d.metadata for d in docs],
    }
