import os, json, glob
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATA_DIR = "data"
INDEX_PATH = "index.faiss"
CHUNKS_PATH = "chunks.json"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

#---------------------------------basic chunking ----------------------------------#

def chunk_text(text:str, chunk_size: int = 1200, overlap:int = 200) -> List[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks

@dataclass
class Chunk:
    id: int
    source_file: str
    text: str
    
def read_txt_files(folder: str) -> List[Tuple[str, str]]:
    paths = sorted(glob.glob(os.path.join(folder, "**/*.txt"), recursive=True))
    files = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            files.append((p, f.read()))
    return files

# --------- embeddings + FAISS ---------
def embed_texts(texts: List[str]) -> np.ndarray:
    # OpenAI embeddings API expects "model", not "mode"
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype=np.float32)

def build_index() -> None:
    files = read_txt_files(DATA_DIR)
    if not files:
        raise RuntimeError(f"No .txt files found in '{DATA_DIR}'")

    chunks: List[Chunk] = []
    for path, content in files:
        for c in chunk_text(content):
            chunks.append(Chunk(id=len(chunks), source_file=os.path.basename(path), text=c))

    print(f"Loaded {len(files)} files → created {len(chunks)} chunks")

    # embed in batches (keeps it safe for rate limits)
    batch = 64
    vectors_list = []
    for i in range(0, len(chunks), batch):
        part = chunks[i:i+batch]
        vecs = embed_texts([x.text for x in part])
        vectors_list.append(vecs)
        print(f"Embedded {min(i+batch, len(chunks))}/{len(chunks)} chunks")

    vectors = np.vstack(vectors_list)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity with normalized vectors
    faiss.normalize_L2(vectors)
    index.add(vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump([chunk.__dict__ for chunk in chunks], f, ensure_ascii=False, indent=2)

    print(f"Saved: {INDEX_PATH}, {CHUNKS_PATH}")

def load_index() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise RuntimeError("Index not found. Run: python app.py build")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def semantic_search(query: str, index: faiss.Index, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q_vec = embed_texts([query])
    faiss.normalize_L2(q_vec)
    scores, ids = index.search(q_vec, TOP_K)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = chunks[idx]
        results.append({
            "score": float(score),
            "source_file": item["source_file"],
            "text": item["text"],
            "chunk_id": item["id"],
        })
    return results

# --------- answering ---------
def build_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for h in hits:
        context_blocks.append(
            f"[source: {h['source_file']} | chunk: {h['chunk_id']} | score: {h['score']:.3f}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    return f"""You answer ONLY using the provided sources.
If the answer is not in the sources, say: "I couldn't find this in the provided files."

Question:
{question}

Sources:
{context}
"""

def answer_question(question: str) -> None:
    index, chunks = load_index()
    hits = semantic_search(question, index, chunks)

    if not hits:
        print("No results found.")
        return

    prompt = build_prompt(question, hits)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful assistant that cites sources."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    print("\n=== Answer ===\n")
    print(resp.choices[0].message.content)

    print("\n=== Top Matches ===")
    for h in hits:
        print(f"- {h['source_file']} (chunk {h['chunk_id']}, score {h['score']:.3f})")

# --------- CLI ---------
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python app.py build")
        print('  python app.py ask "your question"')
        return

    cmd = sys.argv[1].lower()
    if cmd == "build":
        build_index()
    elif cmd == "ask":
        q = " ".join(sys.argv[2:]).strip()
        if not q:
            raise ValueError('Provide a question: python app.py ask "..."')
        answer_question(q)
    else:
        raise ValueError("Unknown command. Use: build | ask")

if __name__ == "__main__":
    main()