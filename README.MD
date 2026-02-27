
# 📄 TXT-RAG-Search

A lightweight **Retrieval-Augmented Generation (RAG)** system that allows you to search and ask questions across a folder of `.txt` files using **semantic search** and **FAISS**.

This project demonstrates how to build a complete local RAG pipeline: document ingestion, chunking, embeddings, vector indexing, retrieval, and grounded answer generation.

---

## 🚀 Features

- Semantic search across multiple `.txt` files  
- Automatic text chunking with overlap  
- FAISS vector index for fast similarity search  
- Answers grounded strictly in retrieved content  
- Source file + chunk references  
- Simple command-line interface  

---

## 🧠 How It Works

1. Load `.txt` files from the `data/` folder  
2. Split text into overlapping chunks  
3. Generate embeddings using OpenAI  
4. Store embeddings in a FAISS vector index  
5. When a question is asked:
   - Embed the question  
   - Retrieve top-K relevant chunks  
   - Send retrieved context + question to the LLM  
   - Generate a grounded answer  

---

## 📁 Project Structure

```

txt-rag-search/
├── data/              # Add your .txt files here
├── app.py             # Main RAG pipeline
├── requirements.txt
├── README.md
├── .env               # OpenAI API key (not committed)
└── .gitignore

````

---

## ⚙️ Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/txt-rag-search.git
cd txt-rag-search
````

---

### 2️⃣ Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add OpenAI API key

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## 📚 Add Data

Place one or more `.txt` files inside the `data/` folder.

Example:

```
data/
├── onboarding.txt
├── product_notes.txt
└── faq.txt
```

---

## 🏗️ Build the Vector Index

Generate embeddings and create the FAISS index:

```bash
python app.py build
```

This creates:

* `index.faiss`
* `chunks.json`

---

## ❓ Ask Questions

```bash
python app.py ask "What is this project about?"
```

The output includes:

* Generated answer
* Source file names
* Chunk IDs
* Similarity scores

---

## 🧪 Example

**Command**

```bash
python app.py ask "What does the onboarding process include?"
```

**Output**

```text
The onboarding process includes account setup, document verification, and initial training.

Top Matches:
- onboarding.txt (chunk 2, score 0.89)
```

---

## 🛠️ Tech Stack

* Python
* OpenAI Embeddings (`text-embedding-3-small`)
* FAISS
* NumPy
* python-dotenv

---

## 📌 Notes

* The index only needs to be rebuilt if the data changes.
* If an answer is not found in the documents, the system explicitly states so.
* Designed as a simple, local RAG pipeline for learning and portfolio use.

---

## 🔮 Future Improvements

* Hybrid search (keyword + semantic)
* Metadata filtering
* FastAPI backend
* Streamlit / React UI
* Persistent vector database (Pinecone / Weaviate)
* Caching and performance optimizations

---

## 🎯 Purpose

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Vector similarity search using FAISS
* Grounding LLM responses in external data
* Practical AI system design

Ideal for **AI, Backend, and Agentic Systems** portfolios.

---

## 📄 License

MIT License

```

---

✅ This is **GitHub-perfect**  
✅ Ready for **Agentic AI Day-2**  
✅ Resume + LinkedIn worthy  

Next (only if you say yes):  
- **Day-2 LinkedIn post**  
- **Resume bullet**  
- **Day-3: PDF RAG → Production-grade API** 🚀
```

