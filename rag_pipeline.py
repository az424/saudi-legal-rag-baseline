# rag_pipeline.py
import os, json, gc
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATA_PATH = "full_systems_dataset_fixed.json"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

# =====================
# Global Variables (Lazy Loading)
# =====================
_faiss_retriever = None
_bm25_retriever = None
client = OpenAI()

def _initialize_retrievers():
    """
    يقوم بتحميل البيانات وبناء الفهارس فقط عند الحاجة لتقليل استهلاك الذاكرة عند الإقلاع.
    """
    global _faiss_retriever, _bm25_retriever

    if _faiss_retriever is not None:
        return _faiss_retriever, _bm25_retriever

    print("[RAG] Loading dataset into memory...")
    
    # 1. Load Data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        page_content = f"System: {item['system']}\nArticle: {item['article_number']}\nContent: {item['text']}"
        metadata = {
            "system": item["system"],
            "article_number": item["article_number"],
            "original_text": item["text"],
            "article_key": item["article_key"] # Important for RRF deduplication
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    print(f"[RAG] Created {len(documents)} documents.")

    # 2. Free raw JSON memory immediately
    del data
    gc.collect()

    # 3. Build Retrievers
    print("[RAG] Building FAISS Index...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    faiss_store = FAISS.from_documents(documents, embeddings)
    _faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 20})

    print("[RAG] Building BM25 Index...")
    _bm25_retriever = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = 20

    # 4. Free documents memory (indexes have their own copy/reference)
    # ملاحظة: BM25 يحتفظ بنسخة، و FAISS يحتفظ بنسخة. 
    # حذف القائمة الأصلية يوفر بعض الذاكرة.
    del documents
    gc.collect()

    print("[RAG] Initialization Complete.")
    return _faiss_retriever, _bm25_retriever

# =====================
# RRF Logic
# =====================
def reciprocal_rank_fusion(results: list[list[Document]], k=60):
    fused_scores = {}
    doc_map = {}
    for source_docs in results:
        for rank, doc in enumerate(source_docs):
            doc_id = doc.metadata.get("article_key")
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank)
    
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, score in reranked]

def hybrid_retrieve(query: str, k=5):
    # Load retrievers on demand
    faiss_retriever, bm25_retriever = _initialize_retrievers()

    bm25_docs = bm25_retriever.invoke(query)
    faiss_docs = faiss_retriever.invoke(query)
    fused_docs = reciprocal_rank_fusion([bm25_docs, faiss_docs])
    return fused_docs[:k]

# =====================
# Prompt Builder
# =====================
def build_prompt(question, docs):
    context = ""
    for i, d in enumerate(docs, 1):
        m = d.metadata
        context += (
            f"--- Document {i} ---\n"
            f"System: {m['system']}\n"
            f"Article: {m['article_number']}\n"
            f"Text: {m['original_text']}\n\n"
        )

    return f"""
You are a Saudi Legal Expert AI. 
Answer the user's question accurately using ONLY the provided legal texts below.

Context:
{context}

User Question: {question}

Guidelines:
1. If the answer exists in the context, quote the Article Number and System Name explicitly.
2. If the context does not contain the answer, say "The provided documents do not contain the answer." Do not make up laws.
3. Answer in Arabic.
""".strip()

# =====================
# Public API Function
# =====================
def answer_question(question: str):
    try:
        docs = hybrid_retrieve(question, k=5)

        if not docs:
            return {"answer": "لم أتمكن من العثور على مراجع مناسبة.", "articles": []}

        prompt = build_prompt(question, docs)

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "articles": [d.metadata for d in docs],
        }
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {"answer": "حدث خطأ أثناء معالجة الطلب.", "articles": []}