# classifier.py
import json
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

_initialized = False

def _init_classifier():
    global _initialized
    global tfidf, svm_clf, embed_model, logreg

    if _initialized:
        return

    print("[Classifier] Initializing (lazy)...")

    with open("full_systems_dataset_fixed.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [x["text"] for x in data]
    labels = [x["system_code"] for x in data]

    tfidf = TfidfVectorizer(max_features=40000, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(texts)
    svm_clf = LinearSVC()
    svm_clf.fit(X_tfidf, labels)

    embed_model = SentenceTransformer("intfloat/e5-small-v2")
    X_emb = embed_model.encode(texts, show_progress_bar=False)
    logreg = LogisticRegression(max_iter=3000)
    logreg.fit(X_emb, labels)

    _initialized = True
    print("[Classifier] Ready.")

def predict_system(query: str) -> str:
    _init_classifier()

    q_tfidf = tfidf.transform([query])
    pred_svm = svm_clf.predict(q_tfidf)[0]
    svm_conf = abs(svm_clf.decision_function(q_tfidf)).max()

    q_emb = embed_model.encode([query])
    pred_emb = logreg.predict(q_emb)[0]
    emb_conf = logreg.predict_proba(q_emb).max()

    return pred_svm if svm_conf > emb_conf else pred_emb
