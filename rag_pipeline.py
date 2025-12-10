# ============================================
# rag_pipeline.py — تحسين RAG مع الاستفادة من الميتاداتا
# ============================================

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

load_dotenv()

# ================================
# 1) إعداد المسارات والنماذج
# ================================
DATA_PATH = os.getenv("LAWS_FILE", "full_systems_dataset_fixed.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# ================================
# 2) تحميل البيانات القانونية
# ================================

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# في حال كان الملف عبارة عن قائمة قوائم
if isinstance(data, list) and data and isinstance(data[0], list):
    flat = []
    for sub in data:
        flat.extend(sub)
    data = flat

documents = []
for item in data:
    if not isinstance(item, dict):
        continue

    system = item.get("system", "").strip()
    article_number = item.get("article_number", "").strip()
    text = item.get("text", "").strip()

    # ⚠️ مهم: هنا نخلي النص المستخدم في BM25/FAISS بسيط وواضح مثل القديم
    page_content = f"{system} - المادة {article_number}:\n{text}"

    # الميتاداتا الكاملة تبقى هنا فقط
    metadata = {
        "system": system,
        "system_code": item.get("system_code", "").strip(),
        "article_key": item.get("article_key", "").strip(),
        "article_number": article_number,
        "original_text": text,
    }

    documents.append(Document(page_content=page_content, metadata=metadata))

# ================================
# 3) بناء الـ Retrievers
# ================================
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vector_store = FAISS.from_documents(documents, embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 8})

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 8


def hybrid_retrieve(query: str, k: int = 5):
    """
    استرجاع هجين: BM25 أولاً (قوي للنص العربي)، ثم FAISS كتعزيز.
    نستخدم article_key (أو system+article_number) كمفتاح فريد لمنع التكرار.
    """
    bm25_docs = bm25_retriever.invoke(query)
    faiss_docs = faiss_retriever.invoke(query)

    merged = []
    seen = set()

    def add_docs(docs):
        for doc in docs:
            meta = doc.metadata or {}
            key = meta.get("article_key") or (meta.get("system"), meta.get("article_number"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= k:
                break

    # نعطي أولوية لـ BM25
    add_docs(bm25_docs)
    # ثم نكمّل من FAISS إذا نحتاج
    if len(merged) < k:
        add_docs(faiss_docs)

    return merged


# ================================
# 4) إعداد الـ Prompt
# ================================
client = OpenAI()


def docs_to_articles_with_refs(docs):
    """
    تجهيز شكل المواد لإرجاعها في الـ API واستخدامها في الـ Prompt.
    """
    articles = []
    refs = []

    for i, doc in enumerate(docs, start=1):
        ref = f"[{i}]"
        meta = doc.metadata or {}

        system = meta.get("system", "")
        system_code = meta.get("system_code", "")
        article_key = meta.get("article_key", "")
        article_number = meta.get("article_number", "")
        text = meta.get("original_text", doc.page_content)

        article = {
            "ref": ref,
            "system": system,
            "system_code": system_code,
            "article_key": article_key,
            "article_number": article_number,
            "text": text,
        }

        articles.append(article)

        refs.append(
            f"{ref} {system}"
            + (f" ({system_code})" if system_code else "")
            + (f" – {article_number}" if article_number else "")
            + (f" – {article_key}" if article_key else "")
        )

    return articles, refs


def build_teacher_prompt(question: str, articles, refs_list):
    """
    الـ prompt المنظم: يستفيد من الميتاداتا لكن دون التأثير على الاسترجاع.
    """
    articles_text = ""
    for a in articles:
        line_header = f"{a['ref']} {a['system']}"
        if a["system_code"]:
            line_header += f" ({a['system_code']})"
        if a["article_number"]:
            line_header += f" - {a['article_number']}"
        if a["article_key"]:
            line_header += f" - {a['article_key']}"

        articles_text += line_header + "\n"
        articles_text += f"النص: {a['text']}\n\n"

    refs_text = "\n".join(refs_list)

    prompt = f"""
أنت مساعد قانوني خبير في الأنظمة السعودية، خصوصًا الأنظمة التجارية والإثبات والمرافعات ولوائحها التنفيذية.

دورك:
1. تعتمد فقط على المواد النظامية المرفقة أدناه.
2. لا تذكر أي حكم لا يوجد له سند صريح في النصوص.
3. عند ذكر حكم، اربطه بالمراجع مثل [1] أو [2].
4. إذا كانت النصوص غير كافية لإجابة جازمة، وضّح ذلك بوضوح.

المواد النظامية المتاحة:
{articles_text}

المراجع:
{refs_text}

السؤال القانوني:
{question}

التعليمات:
- أجب بالعربية الفصحى بأسلوب قانوني منظم ودقيق.
- استخدم فقرات واضحة، وعند الاستشهاد استخدم [1] [2] [3].
- لا تضف مواد غير موجودة في القائمة أعلاه.
"""
    return prompt.strip()


def call_gpt_teacher(prompt: str) -> str:
    """
    استدعاء نموذج GPT للإجابة القانونية المبنية على النصوص المسترجعة.
    """
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


# ================================
# 5) الدالة العامة المستخدمة في الـ API
# ================================
def answer_question(question: str):
    docs = hybrid_retrieve(question, k=5)

    if not docs:
        return {
            "answer": "لم يتم العثور على مواد نظامية مناسبة للإجابة على هذا السؤال.",
            "articles": [],
        }

    articles, refs = docs_to_articles_with_refs(docs)
    prompt = build_teacher_prompt(question, articles, refs)
    answer = call_gpt_teacher(prompt)

    return {
        "answer": answer,
        "articles": articles,
    }
