# rag_pipeline.py

import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ======================
# 1) Load legal dataset
# ======================
with open("full_systems_dataset_fixed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data[0], list):
    data = [entry for sublist in data for entry in sublist]

documents = []
for item in data:
    if not isinstance(item, dict):
        continue

    page_content = f"{item.get('system', '')} - المادة {item.get('article_number', '')}: {item.get('text', '')}"
    metadata = {
        "system": item.get("system", ""),
        "article_number": item.get("article_number", ""),
        "original_text": item.get("text", "")
    }
    documents.append(Document(page_content=page_content, metadata=metadata))

# ======================
# 2) Build retrievers
# ======================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(documents, embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

def hybrid_retrieve(query: str, k: int = 5):
    bm25_docs = bm25_retriever.invoke(query)
    faiss_docs = faiss_retriever.invoke(query)

    merged = []
    seen = set()

    def add_docs(docs):
        for doc in docs:
            key = (doc.metadata.get("system"), doc.metadata.get("article_number"))
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

# ======================
# 3) Prompt + OpenAI client
# ======================
client = OpenAI()  # API key from env: OPENAI_API_KEY

def docs_to_articles_with_refs(docs):
    articles = []
    refs = []
    for i, doc in enumerate(docs, start=1):
        ref = f"[{i}]"
        meta = doc.metadata or {}
        articles.append({
            "ref": ref,
            "system": meta.get("system", ""),
            "article_number": meta.get("article_number", ""),
            "text": meta.get("original_text", doc.page_content),
        })
        refs.append(f"{ref} {meta.get('system', '')} – {meta.get('article_number', '')}")
    return articles, refs

def build_teacher_prompt(question: str, articles, refs_list):
    articles_text = ""
    for a in articles:
        articles_text += f"{a['ref']} {a['system']} – {a['article_number']}\n"
        articles_text += f"النص: {a['text']}\n\n"

    refs_text = "\n".join(refs_list)

    prompt = f"""
أنت مساعد قانوني خبير في الأنظمة السعودية، خصوصًا الأنظمة التجارية والإثبات والمرافعات ولوائحها التنفيذية.

دورك:
1. تعتمد أساسًا على النصوص النظامية المرفقة أدناه.
2. لا تذكر أي حكم غير مدعوم صراحة بنص واضح من المواد المرفقة.
3. عندما تذكر حكمًا، اربطه بالمراجع مثل [1] أو [2].
4. إذا كانت النصوص غير كافية لإجابة جازمة، وضّح ذلك.

المواد النظامية المتاحة:
{articles_text}

المراجع:
{refs_text}

السؤال القانوني:
{question}

التعليمات:
- أجب بالعربية الفصحى بأسلوب قانوني منظم.
- استخدم فقرات واضحة وعند الاستشهاد استخدم [1], [2], [3].
- لا تضف مواد من خارج القائمة أعلاه.

اكتب الجواب الآن:
"""
    return prompt.strip()

def call_gpt_teacher(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()

# ======================
# 4) Public function for API
# ======================
def answer_question(question: str):
    docs = hybrid_retrieve(question, k=5)
    if not docs:
        return {
            "answer": "لم أجد مواد نظامية كافية للإجابة على هذا السؤال.",
            "articles": [],
        }

    articles, refs = docs_to_articles_with_refs(docs)
    prompt = build_teacher_prompt(question, articles, refs)
    answer = call_gpt_teacher(prompt)

    return {
        "answer": answer,
        "articles": articles,
    }
