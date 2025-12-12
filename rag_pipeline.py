# rag_pipeline.py
import os, json, gc
import cohere 
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
COHERE_KEY = os.getenv("COHERE_API_KEY") 

client = OpenAI()
co = None
if COHERE_KEY:
    try:
        co = cohere.Client(COHERE_KEY)
    except:
        print("[RAG] Failed to init Cohere client.")

# =====================
# Global Variables (Lazy Loading)
# =====================
_faiss_retriever = None
_bm25_retriever = None

def _initialize_retrievers():
    global _faiss_retriever, _bm25_retriever
    if _faiss_retriever is not None:
        return _faiss_retriever, _bm25_retriever

    print("[RAG] Loading dataset...")
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("[RAG] ERROR: Dataset file not found!")
        return None, None

    documents = []
    for item in data:
        # دمجنا اسم النظام ورقم المادة في النص لضمان التقاطها في البحث
        full_text = f"النظام: {item['system']}\nالمادة رقم: {item['article_number']}\nالنص: {item['text']}"
        metadata = {
            "system": item["system"],
            "article_number": item["article_number"],
            "original_text": item["text"],
            "article_key": item.get("article_key", str(item["article_number"]))
        }
        documents.append(Document(page_content=full_text, metadata=metadata))
    
    del data
    gc.collect()

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    faiss_store = FAISS.from_documents(documents, embeddings)
    _faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 30}) # وسعنا النطاق

    _bm25_retriever = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = 30 

    del documents
    gc.collect()
    
    print("[RAG] Ready.")
    return _faiss_retriever, _bm25_retriever

# =====================
# The "Smart" Layer: Query Expansion
# =====================
def optimize_query_for_legal_search(user_query: str):
    """
    تحويل سؤال المستخدم العامي إلى مصطلحات قانونية دقيقة للبحث.
    مثال: "مديري طردني" -> "إنهاء عقد العمل الفصل التعسفي نظام العمل المادة 77"
    """
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "أنت خبير قانوني. قم بصياغة سؤال المستخدم ليحتوي على المصطلحات القانونية السعودية الصحيحة (مثل: نظام العمل، الأحوال الشخصية) والكلمات المفتاحية للبحث. أخرج نص البحث فقط."},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3,
            max_tokens=100
        )
        optimized = response.choices[0].message.content.strip()
        print(f"[Query Transform] Original: {user_query} -> Optimized: {optimized}")
        return optimized
    except:
        return user_query

# =====================
# Retrieval Logic
# =====================
def reciprocal_rank_fusion(results: list[list[Document]], k=60):
    fused_scores = {}
    doc_map = {}
    for source_docs in results:
        for rank, doc in enumerate(source_docs):
            # استخدام النص كمعرف فريد في حال غياب المفتاح
            doc_id = doc.metadata.get("article_key", doc.page_content[:20])
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1 / (k + rank)
    
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[doc_id] for doc_id, score in reranked]

def get_relevant_docs(query: str):
    # 1. تحسين الاستعلام (الخطوة السحرية)
    legal_query = optimize_query_for_legal_search(query)
    
    # 2. البحث المختلط
    faiss_retriever, bm25_retriever = _initialize_retrievers()
    if not faiss_retriever: return []

    bm25_docs = bm25_retriever.invoke(legal_query) # نبحث بالمصطلحات القانونية
    faiss_docs = faiss_retriever.invoke(query)     # نبحث بالمعنى الأصلي أيضاً
    
    # 3. دمج النتائج
    broad_docs = reciprocal_rank_fusion([bm25_docs, faiss_docs])[:20]

    # 4. إعادة الترتيب بذكاء (Rerank)
    if co and broad_docs:
        try:
            rerank_resp = co.rerank(
                model="rerank-multilingual-v3.0",
                query=query, # نعيد الترتيب بناء على سؤال المستخدم الأصلي
                documents=[d.page_content for d in broad_docs],
                top_n=5
            )
            final_docs = [broad_docs[r.index] for r in rerank_resp.results]
            return final_docs
        except Exception as e:
            print(f"[RAG] Cohere Error: {e}")
            return broad_docs[:5]
    
    return broad_docs[:5]

# =====================
# Final Answer Generation
# =====================
def answer_question(question: str):
    docs = get_relevant_docs(question)
    
    if not docs:
        return {"answer": "عذراً، لم أجد نصوصاً قانونية ذات صلة في قاعدة البيانات.", "articles": []}

    context_str = "\n\n".join([
        f"--- المستند {i+1} ---\n{d.page_content}" 
        for i, d in enumerate(docs)
    ])

    prompt = f"""
أنت مستشار قانوني سعودي.
أجب على السؤال التالي بناءً **فقط** على النصوص القانونية المقدمة أدناه.

السياق القانوني:
{context_str}

السؤال: {question}

التعليمات:
1. اذكر اسم النظام ورقم المادة بوضوح في إجابتك.
2. كن مباشراً ودقيقاً.
3. إذا لم تجد الإجابة في السياق، قل "لا توجد معلومات كافية في المصادر المتوفرة".
"""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "articles": [d.metadata for d in docs]
    }