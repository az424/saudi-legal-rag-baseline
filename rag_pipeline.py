# rag_pipeline.py

import os
import json
import time
import re
import requests

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings

from openai import OpenAI

load_dotenv()

# ==========================
# 0) إعداد المتغيرات العامة
# ==========================

MODEL_BACKEND = os.getenv("MODEL_BACKEND", "openai").lower()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
HF_TOKEN = os.getenv("HF_TOKEN")

ALLAM_MODEL_ID = os.getenv("ALLAM_MODEL_ID", "SDAIA-ALLEM/allam-2-9b-instruct")
JAIS_MODEL_ID = os.getenv("JAIS_MODEL_ID", "inceptionai/jais-30b-v3")

# عميل OpenAI (يُستخدم فقط إذا كان MODEL_BACKEND = openai)
openai_client = OpenAI()  # يعتمد على OPENAI_API_KEY في الـ .env


# ==========================
# 1) دوال مساعدة عامة
# ==========================

AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
EN_DIGITS = "0123456789"
DIGIT_TRANS = str.maketrans(AR_DIGITS, EN_DIGITS)


def normalize_numbers(text: str) -> str:
    """تحويل الأرقام العربية إلى إنجليزية داخل النص (مثل ٩٠/٢ → 90/2)."""
    if not isinstance(text, str):
        return text
    return text.translate(DIGIT_TRANS)


def debug_print(msg: str):
    """لطباعة معلومات وقت التنفيذ (تقدر تعطلها لاحقًا)."""
    print(f"[RAG-DEBUG] {msg}")


# ==========================
# 2) إعداد Embeddings عربية
# ==========================

def load_arabic_embeddings():
    """
    نستخدم نموذج multilingual-e5-large
    وهو قوي جدًا في البحث الدلالي متعدد اللغات (ومنها العربية).
    """
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


embeddings = load_arabic_embeddings()


# ==========================
# 3) تحميل البيانات القانونية
# ==========================

with open("full_systems_dataset_fixed.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# في بعض الحالات يكون البيانات على شكل قائمة قوائم
if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], list):
    data = [item for sub in raw_data for item in sub]
else:
    data = raw_data

documents = []
for item in data:
    if not isinstance(item, dict):
        continue

    system = item.get("system", "")
    article_number = normalize_numbers(str(item.get("article_number", "")))
    text = item.get("text", "")

    page_content = f"{system} - المادة {article_number}: {text}"
    metadata = {
        "system": system,
        "article_number": article_number,
        "original_text": text,
    }
    documents.append(Document(page_content=page_content, metadata=metadata))

debug_print(f"Loaded {len(documents)} documents into RAG.")


# ==========================
# 4) بناء FAISS + BM25
# ==========================

# FAISS (dense retriever)
vector_store = FAISS.from_documents(documents, embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# BM25 (sparse retriever)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10  # نأخذ 10 مبدئيًا من كل واحد


def hybrid_retrieve(query: str, k: int = 5):
    """
    استرجاع هجين:
    - BM25 (وزن 2.0)
    - FAISS/Embeddings (وزن 1.0)
    ونختار أعلى k مواد بعد الدمج.
    """
    t0 = time.time()
    query_norm = normalize_numbers(query)

    docs_bm25 = bm25_retriever.get_relevant_documents(query_norm)
    docs_faiss = faiss_retriever.get_relevant_documents(query_norm)

    combined = {}

    # إعطاء وزن أعلى لـ BM25 لأنه ممتاز للنصوص النظامية
    for d in docs_bm25:
        key = (d.metadata.get("system"), d.metadata.get("article_number"))
        if key not in combined:
            combined[key] = (d, 2.0)

    # Embeddings كدعم دلالي
    for d in docs_faiss:
        key = (d.metadata.get("system"), d.metadata.get("article_number"))
        if key not in combined:
            combined[key] = (d, 1.0)

    # ترتيب حسب الوزن (تنازليًا)
    sorted_docs = sorted(combined.values(), key=lambda x: -x[1])
    top_docs = [d[0] for d in sorted_docs][:k]

    debug_print(
        f"Hybrid retrieve: BM25={len(docs_bm25)}, FAISS={len(docs_faiss)}, "
        f"merged={len(sorted_docs)}, returned={len(top_docs)}, "
        f"time={time.time() - t0:.3f}s"
    )

    return top_docs


# ==========================
# 5) تحويل المستندات إلى مواد + مراجع
# ==========================

def docs_to_articles_with_refs(docs):
    """
    نحول مستندات LangChain إلى قائمة مواد + مراجع نصية لاستخدامها في الـ prompt
    """
    articles = []
    refs = []
    for i, doc in enumerate(docs, start=1):
        ref = f"[{i}]"
        meta = doc.metadata or {}
        system = meta.get("system", "")
        article_number = meta.get("article_number", "")
        text = meta.get("original_text", doc.page_content)

        articles.append(
            {
                "ref": ref,
                "system": system,
                "article_number": article_number,
                "text": text,
            }
        )
        refs.append(f"{ref} {system} – {article_number}")
    return articles, refs


# ==========================
# 6) بناء الـ Prompt القانوني
# ==========================

def build_teacher_prompt(question: str, articles, refs_list):
    articles_text = ""
    for a in articles:
        articles_text += f"{a['ref']} {a['system']} – {a['article_number']}\n"
        articles_text += f"النص:\n{a['text']}\n\n"

    refs_text = "\n".join(refs_list)

    prompt = f"""
أنت نظام قانوني متخصص في الأنظمة السعودية (التجارية، المرافعات، الإثبات، الإفلاس ولوائحها التنفيذية).

المواد النظامية المتاحة أمامك (لا تستخدم أي شيء من خارجها):

{articles_text}

المراجع المتاحة:
{refs_text}

السؤال القانوني:
{question}

التعليمات:
- أجب بالعربية الفصحى وبأسلوب قانوني منظم.
- استند فقط إلى المواد المذكورة أعلاه.
- عندما تستند إلى مادة، استخدم رقم المرجع بين قوسين مثل [1] أو [2].
- إذا كانت المواد غير كافية لإجابة جازمة، وضّح ذلك صراحة.

اكتب الإجابة القانونية الآن:
"""
    return prompt.strip()


# ==========================
# 7) دوال استدعاء الـ LLMs
# ==========================

def call_openai_teacher(prompt: str) -> str:
    """استدعاء نموذج OpenAI (مثلاً GPT-4.1)"""
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    return resp.choices[0].message.content.strip()


def _call_hf_text_gen(prompt: str, model_id: str) -> str:
    """
    استدعاء نموذج توليد نصوص عن طريق Hugging Face Inference API
    (يستخدم مع Allam / Jais).
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in environment variables.")

    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 2000,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # تنسيقات شائعة لنتائج HF
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            text = first.get("generated_text") or first.get("summary_text")
            if isinstance(text, str):
                return text.strip()

    if isinstance(data, dict):
        text = data.get("generated_text") or data.get("summary_text")
        if isinstance(text, str):
            return text.strip()

    raise RuntimeError(f"Unexpected HF response: {data}")


def call_allam_teacher(prompt: str) -> str:
    """استدعاء نموذج Allam عبر HF API"""
    return _call_hf_text_gen(prompt, ALLAM_MODEL_ID)


def call_jais_teacher(prompt: str) -> str:
    """استدعاء نموذج Jais عبر HF API"""
    return _call_hf_text_gen(prompt, JAIS_MODEL_ID)


def call_teacher_llm(prompt: str) -> str:
    """
    يختار النموذج المناسب بناءً على MODEL_BACKEND:
    - openai
    - allam
    - jais
    """
    backend = (MODEL_BACKEND or "openai").lower()
    debug_print(f"Using backend: {backend}")

    if backend == "allam":
        return call_allam_teacher(prompt)
    if backend == "jais":
        return call_jais_teacher(prompt)
    # الافتراضي
    return call_openai_teacher(prompt)


# ==========================
# 8) الدالة التي يستدعيها API.py
# ==========================

def answer_question(question: str):
    """
    هذه هي الدالة العامة التي يستدعيها API.py:
    - تسترجع المواد
    - تبني prompt
    - تستدعي LLM
    - تعيد الإجابة + المواد المرتبطة
    """
    docs = hybrid_retrieve(question, k=5)

    if not docs:
        return {
            "answer": "لم أجد مواد نظامية كافية للإجابة على هذا السؤال.",
            "articles": [],
        }

    articles, refs = docs_to_articles_with_refs(docs)
    prompt = build_teacher_prompt(question, articles, refs)
    answer = call_teacher_llm(prompt)

    return {
        "answer": answer,
        "articles": articles,
    }
