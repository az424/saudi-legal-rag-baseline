# rag_pipeline.py
import os, json, gc
# يجب التأكد من تثبيت مكتبة cohere: pip install cohere
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
COHERE_KEY = os.getenv("COHERE_API_KEY") # تأكد من إضافة هذا المتغير في Render

# =====================
# Global Variables (Lazy Loading)
# =====================
_faiss_retriever = None
_bm25_retriever = None
client = OpenAI()
co = None # Cohere client placeholder

def _initialize_retrievers():
    global _faiss_retriever, _bm25_retriever, co
    if _faiss_retriever is not None:
        return _faiss_retriever, _bm25_retriever

    print("[RAG] Initializing... Loading dataset...")
    
    # Initialize Cohere client if key exists
    if COHERE_KEY:
        co = cohere.Client(COHERE_KEY)
        print("[RAG] Cohere Reranker initialized.")
    else:
        print("[RAG] WARNING: COHERE_API_KEY not found. Reranking will be skipped.")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        # تحسين المحتوى ليكون غنياً بالمعلومات للمقيم (Reranker)
        page_content = f"نظام: {item['system']}\nرقم المادة: {item['article_number']}\nنص المادة: {item['text']}"
        metadata = {
            "system": item["system"],
            "article_number": item["article_number"],
            "original_text": item["text"],
            "article_key": item["article_key"]
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    del data
    gc.collect()

    # زيادة عدد النتائج الأولية في الفهارس
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    faiss_store = FAISS.from_documents(documents, embeddings)
    _faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 30}) # نطلب 30 مبدئياً

    _bm25_retriever = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = 30 # نطلب 30 مبدئياً

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
    # نعيد أفضل 20 نتيجة هنا لتمريرها لمرحلة إعادة الترتيب
    return [doc_map[doc_id] for doc_id, score in reranked][:20]

def hybrid_retrieve_broad(query: str):
    """يسترجع مجموعة واسعة من النتائج المرشحة"""
    faiss_retriever, bm25_retriever = _initialize_retrievers()
    bm25_docs = bm25_retriever.invoke(query)
    faiss_docs = faiss_retriever.invoke(query)
    # دمج مبدئي للحصول على أفضل 20 مرشح
    broad_docs = reciprocal_rank_fusion([bm25_docs, faiss_docs])
    return broad_docs

# =====================
# Cohere Reranking Step (The "Pro" Upgrade)
# =====================
def rerank_documents(query: str, docs: list[Document], top_n=5):
    """ يستخدم Cohere لقراءة النتائج وإعادة ترتيبها حسب الصلة الدقيقة """
    if not co or not docs:
        return docs[:top_n] # الرجوع للترتيب العادي إذا لم يعمل Cohere

    print(f"[RAG] Reranking {len(docs)} documents with Cohere...")
    
    # تحضير النصوص لـ Cohere
    docs_content = [d.page_content for d in docs]
    
    try:
        # استخدام النموذج متعدد اللغات الأحدث
        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=docs_content,
            top_n=top_n,
            return_documents=False # نحتاج فقط المؤشرات والدرجات
        )
        
        reranked_docs = []
        for result in response.results:
            # result.index هو ترتيب المستند في القائمة الأصلية
            doc = docs[result.index]
            # نضيف درجة الصلة للميتاداتا (اختياري، للمراقبة)
            doc.metadata["relevance_score"] = result.relevance_score
            reranked_docs.append(doc)
            
        print(f"[RAG] Reranking complete. Top score: {response.results[0].relevance_score}")
        return reranked_docs

    except Exception as e:
        print(f"[RAG] Reranking Failed: {e}. Falling back to original order.")
        return docs[:top_n]

# =====================
# Prompt Builder
# =====================
def build_prompt(question, docs):
    context = ""
    for i, d in enumerate(docs, 1):
        m = d.metadata
        context += (
            f"--- المستند {i} (الصلة: {m.get('relevance_score', 'N/A'):.2f}) ---\n"
            f"النظام: {m['system']}\n"
            f"رقم المادة: {m['article_number']}\n"
            f"النص: {m['original_text']}\n\n"
        )

    return f"""
أنت خبير قانوني سعودي متخصص بالذكاء الاصطناعي.
مهمتك هي الإجابة على سؤال المستخدم بدقة، معتمداً **حصرياً** على النصوص القانونية المقدمة أدناه.

السياق القانوني (مرتب حسب الصلة):
{context}

سؤال المستخدم: {question}

الإرشادات:
1. استخرج الإجابة مباشرة من النصوص المقدمة.
2. يجب أن تذكر صراحة "اسم النظام" و "رقم المادة" التي استندت إليها في إجابتك.
3. إذا كانت النصوص المقدمة لا تحتوي على الإجابة القاطعة، قل: "عذراً، النصوص القانونية المتاحة لا تحتوي على إجابة دقيقة لهذا السؤال." ولا تحاول تأليف إجابة من خارج السياق.
4. أجب باللغة العربية بأسلوب مهني وواضح.
""".strip()

# =====================
# Public API Function
# =====================
def answer_question(question: str):
    try:
        # 1. استرجاع واسع (أفضل 20)
        broad_docs = hybrid_retrieve_broad(question)

        if not broad_docs:
             return {"answer": "لم أتمكن من العثور على مراجع مناسبة في قاعدة البيانات.", "articles": []}

        # 2. إعادة الترتيب الذكي (أفضل 5) - الخطوة الجديدة
        final_docs = rerank_documents(question, broad_docs, top_n=5)

        # 3. بناء السياق والإجابة
        prompt = build_prompt(question, final_docs)

        print("[RAG] Sending to GPT-4...")
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, # الحفاظ على الدقة
            max_tokens=1500,
        )

        return {
            "answer": response.choices[0].message.content.strip(),
            "articles": [d.metadata for d in final_docs],
        }
    except Exception as e:
        print(f"Error generating answer: {e}")
        # طباعة الخطأ كاملاً في السيرفر للمساعدة في التتبع
        import traceback
        traceback.print_exc()
        return {"answer": "حدث خطأ فني أثناء معالجة الطلب.", "articles": []}