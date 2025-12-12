# rag_pipeline.py
import os
import json
import gc
from dotenv import load_dotenv
from openai import OpenAI
import cohere

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

# =====================
# Environment
# =====================
load_dotenv()

DATA_PATH = "full_systems_dataset_fixed.json"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
COHERE_KEY = os.getenv("COHERE_API_KEY")

client = OpenAI()

co = None
if COHERE_KEY:
    try:
        co = cohere.Client(COHERE_KEY)
    except Exception:
        print("[RAG] Cohere init failed.")

# =====================
# Global (Lazy Load)
# =====================
_faiss_retriever = None
_bm25_retriever = None

# =====================
# Dataset & Retrievers
# =====================
def initialize_retrievers():
    global _faiss_retriever, _bm25_retriever
    if _faiss_retriever is not None:
        return _faiss_retriever, _bm25_retriever

    print("[RAG] Loading dataset...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        full_text = (
            f"النظام: {item['system']}\n"
            f"رقم المادة: {item['article_number']}\n"
            f"النص: {item['text']}"
        )
        metadata = {
            "system": item["system"],
            "article_number": item["article_number"],
            "article_key": item.get("article_key", item["article_number"]),
        }
        documents.append(Document(page_content=full_text, metadata=metadata))

    del data
    gc.collect()

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    faiss_store = FAISS.from_documents(documents, embeddings)

    _faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 30})
    _bm25_retriever = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = 30

    del documents
    gc.collect()

    print("[RAG] Retrievers ready.")
    return _faiss_retriever, _bm25_retriever

# =====================
# Query Optimization
# =====================
def optimize_query_for_legal_search(query: str) -> str:
    """
    Force retrieval of governing procedural rules
    """
    return f"""
{query}
تحرير الدعوى
عدم قبول الدعوى
القاعدة الإجرائية العامة
نظام المرافعات الشرعية
المادة 66
"""

# =====================
# Prompt Builder (CRITICAL)
# =====================
def build_prompt(question: str, docs: list[Document]) -> str:
    context = "\n\n".join(
        f"--- المرجع {i+1} ---\n{d.page_content}"
        for i, d in enumerate(docs)
    )

    return f"""
أنت قاضٍ تجاري سعودي.

مهمتك هي الإجابة على السؤال التالي **حصريًا** بناءً على النصوص النظامية المرفقة أدناه، دون أي اجتهاد خارجي.

النصوص النظامية:
{context}

السؤال:
{question}

⚠️ تعليمات إلزامية:
1. ميّز بين القاعدة النظامية العامة (النظام) والأحكام الخاصة أو التنفيذية (اللائحة).
2. لا تطبق نصًا خاصًا على مسألة تحكمها قاعدة عامة.
3. إذا كانت المسألة تتعلق بتحرير الدعوى، فلا تطبق نصوص "ارتباط الطلبات".
4. إذا لم تتضمن النصوص المرفقة القاعدة العامة الحاكمة، **يجب أن تصرّح بعدم كفاية النصوص**.
5. لا تدخل في موضوع الدعوى إذا كان العيب شكليًا.

صيغة الإجابة:
- الحكم النظامي (يجوز / لا يجوز)
- الأثر النظامي (عدم قبول / صرف نظر)
- ذكر المادة النظامية الصحيحة فقط
""".strip()

# =====================
# Validation Layer
# =====================
def has_governing_rule(docs: list[Document]) -> bool:
    """
    Ensure presence of general procedural law
    """
    for d in docs:
        if "نظام المرافعات" in d.metadata.get("system", ""):
            return True
    return False

# =====================
# Retrieval Logic
# =====================
def reciprocal_rank_fusion(results, k=60):
    scores = {}
    doc_map = {}

    for source_docs in results:
        for rank, doc in enumerate(source_docs):
            doc_id = doc.metadata.get("article_key", doc.page_content[:20])
            if doc_id not in scores:
                scores[doc_id] = 0
                doc_map[doc_id] = doc
            scores[doc_id] += 1 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, _ in ranked]

def get_relevant_docs(query: str) -> list[Document]:
    faiss_retriever, bm25_retriever = initialize_retrievers()

    legal_query = optimize_query_for_legal_search(query)

    bm25_docs = bm25_retriever.invoke(legal_query)
    faiss_docs = faiss_retriever.invoke(query)

    fused_docs = reciprocal_rank_fusion([bm25_docs, faiss_docs])[:20]

    if co and fused_docs:
        try:
            rerank = co.rerank(
                model="rerank-multilingual-v3.0",
                query=query,
                documents=[d.page_content for d in fused_docs],
                top_n=5
            )
            return [fused_docs[r.index] for r in rerank.results]
        except Exception:
            return fused_docs[:5]

    return fused_docs[:5]

# =====================
# Answer Generation
# =====================
def answer_question(question: str) -> dict:
    docs = get_relevant_docs(question)

    if not docs:
        return {
            "answer": "لا توجد نصوص نظامية ذات صلة في قاعدة البيانات.",
            "articles": []
        }

    # 🔒 Fail-safe: no governing rule → no answer
    if not has_governing_rule(docs):
        return {
            "answer": (
                "النصوص المرفقة لا تتضمن القاعدة الإجرائية العامة الحاكمة للمسألة "
                "(مثل المادة 66 من نظام المرافعات الشرعية)، "
                "والمواد المتاحة تتعلق بحالات خاصة لا تكفي للفصل في السؤال."
            ),
            "articles": [d.metadata for d in docs]
        }

    prompt = build_prompt(question, docs)

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "articles": [d.metadata for d in docs]
    }
