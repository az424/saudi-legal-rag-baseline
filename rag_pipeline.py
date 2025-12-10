# ============================================
# rag_pipeline.py (Updated, Metadata-Enhanced)
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
# 1) Load legal dataset (UPDATED)
# ================================
DATA_PATH = "full_systems_dataset_fixed.json"   # ← change filename here

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# flatten if needed
if isinstance(data[0], list):
    data = [entry for sublist in data for entry in sublist]

documents = []
for item in data:
    if not isinstance(item, dict):
        continue

    # Build page_content (used for BM25 & FAISS)
    page_content = (
        f"{item.get('system', '')} "
        f"({item.get('system_code', '')}) "
        f"- {item.get('article_number', '')}:\n"
        f"{item.get('text', '')}"
    )

    # Store full metadata so RAG + API can use it
    metadata = {
        "system": item.get("system", ""),
        "system_code": item.get("system_code", ""),
        "article_key": item.get("article_key", ""),
        "article_number": item.get("article_number", ""),
        "original_text": item.get("text", "")
    }

    documents.append(Document(page_content=page_content, metadata=metadata))

# ================================
# 2) Build retrievers (UPDATED)
# ================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = FAISS.from_documents(documents, embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 8})

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 8

def hybrid_retrieve(query: str, k: int = 5):
    """Hybrid BM25 + FAISS with stable dedup by article_key."""
    bm25_docs = bm25_retriever.invoke(query)
    faiss_docs = faiss_retriever.invoke(query)

    merged = []
    seen = set()

    def add_docs(docs):
        for doc in docs:
            meta = doc.metadata
            key = meta.get("article_key") or (meta.get("system"), meta.get("article_number"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
            if len(merged) >= k:
                break

    add_docs(bm25_docs)
    if len(merged) < k:
        add_docs(faiss_docs)

    return merged

# ================================
# 3) Prompt Construction (UPDATED)
# ================================
client = OpenAI()

def docs_to_articles_with_refs(docs):
    """Return clean article structures + reference lines."""
    articles = []
    refs = []

    for i, doc in enumerate(docs, start=1):
        ref = f"[{i}]"
        meta = doc.metadata or {}

        article = {
            "ref": ref,
            "system": meta.get("system", ""),
            "system_code": meta.get("system_code", ""),
            "article_key": meta.get("article_key", ""),
            "article_number": meta.get("article_number", ""),
            "text": meta.get("original_text", doc.page_content),
        }

        articles.append(article)

        refs.append(
            f"{ref} {article['system']} ({article['system_code']}) – "
            f"{article['article_number']} – {article['article_key']}"
        )

    return articles, refs


def build_teacher_prompt(question: str, articles, refs_list):
    """Builds the full legal RAG prompt with full metadata."""
    articles_text = ""

    for a in articles:
        articles_text += (
            f"{a['ref']} {a['system']} ({a['system_code']}) "
            f"- {a['article_number']} - {a['article_key']}\n"
            f"النص: {a['text']}\n\n"
        )

    refs_text = "\n".join(refs_list)

    prompt = f"""
أنت مساعد قانوني خبير في الأنظمة السعودية، خصوصًا الأنظمة التجارية والإثبات والمرافعات ولوائحها التنفيذية.

دورك:
1. تعتمد فقط على المواد النظامية المرفقة أدناه.
2. لا تستنتج أحكامًا غير موجودة في النصوص.
3. عند ذكر حكم، اربطه بالمراجع مثل [1] أو [2].
4. إذا كانت النصوص غير حاسمة، اذكر ذلك بوضوح.

المواد النظامية المتاحة:
{articles_text}

المراجع:
{refs_text}

السؤال:
{question}

التعليمات:
- أجب بالعربية الفصحى بأسلوب قانوني دقيق.
- استخدم الاستشهاد بالأرقام [1] [2] [3].
- لا تضف أي مادة غير موجودة في القائمة.

اكتب الجواب الآن:
"""
    return prompt.strip()


def call_gpt_teacher(prompt: str) -> str:
    """Query GPT model for grounded legal reasoning."""
    response = client.chat.completions.create(
        model="gpt-4.1",   # change if needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()

# ================================
# 4) Public API function (UPDATED)
# ================================
def answer_question(question: str):
    docs = hybrid_retrieve(question, k=5)

    if not docs:
        return {
            "answer": "لم يتم العثور على مواد نظامية مناسبة للإجابة.",
            "articles": [],
        }

    articles, refs = docs_to_articles_with_refs(docs)
    prompt = build_teacher_prompt(question, articles, refs)
    answer = call_gpt_teacher(prompt)

    return {
        "answer": answer,
        "articles": articles,
    }
