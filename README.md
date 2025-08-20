# 📄 RAG-based PDF Analyzer

A **Retrieval-Augmented Generation (RAG) pipeline** for querying unstructured PDF documents. The system combines **document parsing, embeddings, vector database retrieval, and LLM-based answering**, all wrapped in an easy-to-use **Streamlit UI**.

---

## ✨ Features

* 📑 **PDF Ingestion** – Extracts and chunks text from uploaded PDF documents.
* 🔍 **Semantic Search** – Embeds document chunks with **SentenceTransformers** and stores them in **ChromaDB**.
* 🏆 **Cross-Encoder Reranking** – Improves retrieval precision by reranking top candidates.
* 🤖 **Context-Aware Answers** – Uses LLMs (local or API-based) to generate responses grounded in document context.
* ⚡ **GPU Optimized** – Supports running embeddings and reranking models on GPU (FP16 precision, batch size tuning).
* 🎛️ **Config-Driven Architecture** – Easily switch between embedding backends (Hugging Face, Ollama, OpenAI).
* 🌐 **Interactive UI** – Upload PDFs and ask natural language questions via **Streamlit**.

---

## 🏗️ Architecture

```
PDF Upload → Text Chunking → Embedding (SentenceTransformers) → 
ChromaDB (Vector Store) → Retriever → Reranker (CrossEncoder) → 
LLM (Answer Generator) → Streamlit UI
```

---

## ⚙️ Tech Stack

* **Python** (3.9+)
* **PyTorch** – Deep learning backend
* **SentenceTransformers** – Embedding models
* **ChromaDB** – Vector store for semantic retrieval
* **Transformers** – Cross-encoder and reranking models
* **Streamlit** – UI for PDF upload & querying
* **Ollama / OpenAI (Optional)** – LLM integration

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shub1709/rag-based-pdf-analyzer.git
cd rag-based-pdf-analyzer
```

### 2️⃣ Create Virtual Environment

```bash
conda create -n rag_pipeline python=3.9
conda activate rag_pipeline
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure

Update `config.yaml` with your preferences:

```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
cross_encoder: cross-encoder/ms-marco-MiniLM-L-6-v2
llm_backend: openai   # options: openai / ollama / hf
openai_api_key: "your_api_key_here"
```

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 🖼️ Demo

Upload any PDF, ask questions like:

* *“What is the main conclusion of this report?”*
* *“List all mentioned stakeholders.”*
* *“Summarize section 3 in 2 sentences.”*

---

## 📊 Performance Considerations

* Default embedding model: **MiniLM (lightweight, \~22M params)**
* For higher accuracy: try **MPNet** or **bge-large-en** (larger, VRAM-heavy).
* Use **batch\_size=8/16** for GPU memory efficiency.
* Cross-encoder reranking is optional (trade-off: better relevance vs. slower inference).

---

## 📂 Project Structure

```
rag-based-pdf-analyzer/
│── app.py                  # Streamlit entry point
│── src/                    # Core RAG pipeline (retriever, reranker, generator)
    │── rag_pipeline.py
    │── llm_manager.py     
    │── document_processor.py     
    │── embeddigns_manager.py     
    │── retriever.py     
    │── utils.py     
│── config.yaml             # Config file for models, API keys, parameters
│── requirements.txt        # Python dependencies
│── utils/                  # Helpers (PDF parsing, chunking, etc.)
```

---

## 🔮 Future Enhancements

* [ ] Multi-PDF support
* [ ] Metadata-aware search (e.g., by section/page)
* [ ] Caching frequent queries
* [ ] Dockerized deployment

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a PR or raise an issue in the repo.

---

## 📜 License

This project is licensed under the MIT License.