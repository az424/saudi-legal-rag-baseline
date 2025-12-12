# rag_pipeline.py
import os, json
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings

# Note: We removed the classifier import to stop the "Hard Filter" issue.
# If you want to use the classifier, use it to ADD context to the prompt, not to filter documents.
# from classifier import predict_system 

load_dotenv()

DATA_PATH = "full_systems_dataset_fixed.json"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo") # Ensure you use a valid model name

# =====================
# Load Legal Documents
# =====================
print("Loading dataset...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []
for item in data:
    # Adding more context to the page_content improves semantic search
    page_content = f"System: {item['system']}\nArticle: {item['article_number']}\nContent: {item['text']}"
    
    metadata = {
        "system": item["system"],
        "system_code": item["system_code"],
        "article_key": item["article_key"],
        "article_number": item["article_number"],
        "original_text": item["text"],
    }
    documents.append(Document(page_content=page_content, metadata=metadata))

print(f"Loaded {len(documents)} documents.")

# =====================
# Retrievers
# =====================
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
faiss_store = FAISS.from_documents(documents, embeddings)
# Increase k here to cast a wider net initially
faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 20}) 

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 20

# =====================
# RRF (Reciprocal Rank Fusion) Logic
# =====================
def reciprocal_rank_fusion(results: list[list[Document]], k=60):
    """
    Combines results from multiple retrievers using RRF.
    Higher score = better match.
    """
    fused_scores = {}
    doc_map = {}

    for source_docs in results:
        for rank, doc in enumerate(source_docs):
            # Use article_key as a unique identifier to merge duplicates
            doc_id = doc.metadata.get("article_key") 
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            
            # The standard RRF formula: 1 / (k + rank)
            fused_scores[doc_id] += 1 / (k + rank)

    # Sort documents by final score (descending)
    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return the Document objects
    return [doc_map[doc_id] for doc_id, score in reranked]

def hybrid_retrieve(query: str, k=5):
    # 1. Retrieve broad results from both methods
    bm25_docs = bm25_retriever.invoke(query)
    faiss_docs = faiss_retriever.invoke(query)

    # 2. Fuse them intelligently using RRF
    # This ensures that if a doc appears in BOTH lists, it jumps to the top.
    fused_docs = reciprocal_rank_fusion([bm25_docs, faiss_docs])

    # 3. Return top k
    return fused_docs[:k]

# =====================
# Prompt Builder
# =====================
client = OpenAI()

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
    # We removed predict_system to prevent the "Hard Filter" error.
    # If the system is not predicting correctly, it hurts more than it helps.
    
    print(f"[RAG] Processing: {question}")
    
    # Retrieve
    docs = hybrid_retrieve(question, k=5)

    if not docs:
        return {"answer": "لم أتمكن من العثور على مراجع مناسبة.", "articles": []}

    prompt = build_prompt(question, docs)

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, # Lower temperature for strictly factual answers
        max_tokens=2000,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "articles": [d.metadata for d in docs],
    }