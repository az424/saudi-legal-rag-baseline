## ⚠️ Disclaimer

This project and its generated datasets are intended **for research and educational purposes only**.  
The model-generated legal questions, answers, and citations **may contain inaccuracies or errors** and must not be treated as legal advice.  
Please verify all information using **official Saudi legal sources**.  

The author assumes **no responsibility** for any use or misuse of this project.


## Saudi Legal Hybrid RAG

Hybrid Retrieval-Augmented Generation system for Saudi laws.

### Architecture
- Hybrid BM25 + FAISS retrieval
- System-level router using TF-IDF + embeddings
- Conservative legal prompting with mandatory citation

### Usage
```python
from rag_pipeline import answer_question
answer_question("هل يجوز زيادة رأس مال الشركة دون موافقة الشركاء؟")
