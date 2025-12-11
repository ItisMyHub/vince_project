import os
import json
from time import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

# --- CONFIGURATION ---
CHUNKED_FILE = "chunked_corpus.json"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "vince_agent"

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16

def load_chunks(path):
    """Loads chunks and prepares metadata for Chroma."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids, docs, metadatas = [], [], []
    for entry in data:
        ids.append(entry.get("id"))
        docs.append(entry.get("content"))

        meta = {
            "filename": entry.get("filename"),
            "type": entry.get("type"),
            "country": entry.get("country", "unknown"),
            "region": entry.get("region", "unknown"),
            "city": entry.get("city", "unknown"),
            "partner": entry.get("partner", "general"),
            "_model": entry.get("_model", "_local"),
            "_country": entry.get("_country", entry.get("country", "unknown")),
            "_region": entry.get("_region", entry.get("region", "unknown")),
            "_city": entry.get("_city", entry.get("city", "unknown")),
            "_partner": entry.get("_partner", entry.get("partner", "general")),
            "page": str(entry.get("page")) if entry.get("page") else "0"
        }
        metadatas.append(meta)

    return ids, docs, metadatas

def embed_and_store(ids, docs, metadatas):
    print(f"🧠 Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"💾 Initializing ChromaDB at {PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ Deleted existing collection '{COLLECTION_NAME}' to avoid duplicates.")
    except ValueError:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    total = len(docs)
    print(f"🚀 Embedding {total} chunks...")

    start_time = time()
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Embedding"):
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_docs = docs[i:i+BATCH_SIZE]
        batch_meta = metadatas[i:i+BATCH_SIZE]

        embeddings = model.encode(batch_docs, normalize_embeddings=True)

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings.tolist()
        )

    elapsed = time() - start_time
    print(f"✅ Success! Stored {total} vectors in {elapsed:.1f}s.")

def main():
    if not os.path.exists(CHUNKED_FILE):
        print(f"❌ {CHUNKED_FILE} not found. Run corpus_chunker.py first.")
        return

    ids, docs, metadatas = load_chunks(CHUNKED_FILE)
    if not ids:
        print("⚠️ No chunks found.")
        return

    embed_and_store(ids, docs, metadatas)

if __name__ == "__main__":
    main()