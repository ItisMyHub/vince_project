# We are building the Embedding Layer and saving a persistent vector database using Chroma.
# We are using SentenceTransformers to create embeddings and ChromaDB to store them.
# We are keeping the code simple, robust, and batch-friendly for large datasets.

import os
import json
from time import time
from tqdm import tqdm

# We are using SentenceTransformer to compute embeddings on CPU/GPU.
from sentence_transformers import SentenceTransformer

# We are using chromadb to store vectors persistently (duckdb+parquet backend).
import chromadb
from chromadb.config import Settings

# --- CONFIGURATION ---
CHUNKED_FILE = "chunked_corpus.json"   # We are reading the output from corpus_chunker.py
PERSIST_DIR = "./chroma_db"            # We are storing the DB here (will be created if missing)
COLLECTION_NAME = "vince_agent"        # We are naming our collection for later queries

MODEL_NAME = "BAAI/bge-m3"      # We are using this AI model to support multilingual data.
BATCH_SIZE = 64                        # We are embedding texts in batches to manage memory/time

# --- HELPERS / MAIN ---
def load_chunks(path):
    # We are loading the chunked JSON file and returning lists for ids, docs, and metadata.
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []
    docs = []
    metadatas = []

    for entry in data:
        ids.append(entry.get("id"))
        docs.append(entry.get("content"))
        # We are keeping useful metadata for retrieval and debugging.
        metadatas.append({
            "filename": entry.get("filename"),
            "type": entry.get("type"),
            "original_source": entry.get("original_source")
        })

    return ids, docs, metadatas

def ensure_persist_dir(path):
    # We are ensuring the persistence directory exists for Chroma.
    os.makedirs(path, exist_ok=True)

def create_or_reset_collection(client, name):
    # We are checking for an existing collection. If it exists, we will delete and recreate it
    # so that repeated runs overwrite the previous DB (you can change this behavior if desired).
    existing = [c["name"] for c in client.list_collections()]
    if name in existing:
        print(f" Collection '{name}' exists — we are deleting it to create a fresh one.")
        try:
            client.delete_collection(name)
        except Exception as e:
            print(f" Warning: failed to delete existing collection: {e}")
    # We are creating a new empty collection.
    return client.create_collection(name)

def embed_and_store(ids, docs, metadatas):
    # We are preparing the model and the Chroma client, then adding documents batchwise.
    print(" We are loading the embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print(" We are initializing the Chroma client with persistence at:", PERSIST_DIR)
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=PERSIST_DIR
    ))

    # We are creating (or resetting) the collection where vectors will be stored.
    collection = create_or_reset_collection(client, COLLECTION_NAME)

    total = len(docs)
    print(f" We are embedding and storing {total} chunks in batches of {BATCH_SIZE}...")

    start_time = time()
    for i in tqdm(range(0, total, BATCH_SIZE), desc="embedding_batches"):
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_docs = docs[i:i+BATCH_SIZE]
        batch_meta = metadatas[i:i+BATCH_SIZE]

        # We are computing embeddings (returns numpy array).
        embeddings = model.encode(batch_docs, show_progress_bar=False, convert_to_numpy=True)

        # We are adding the batch to Chroma. We pass embeddings directly to avoid using an embedding function.
        try:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=embeddings.tolist()  # chroma accepts nested lists
            )
        except Exception as e:
            print(f" Error adding batch starting at {i}: {e}")
            # We are continuing so that one problematic batch does not stop the whole process.
            continue

    # We are ensuring everything is persisted to disk.
    try:
        client.persist()
    except Exception as e:
        print(f" Warning: client.persist() failed or is unnecessary for this version of chromadb: {e}")

    elapsed = time() - start_time
    print(f" Done. We stored {total} chunks to collection '{COLLECTION_NAME}' in {elapsed:.1f}s.")
    print(f" Chroma DB files are in: {PERSIST_DIR}")

def main():
    # We are validating input file presence first.
    if not os.path.exists(CHUNKED_FILE):
        print(f" We cannot find {CHUNKED_FILE}. Please run corpus_chunker.py first.")
        return

    # We are loading the chunked corpus prepared earlier.
    ids, docs, metadatas = load_chunks(CHUNKED_FILE)
    if not ids:
        print(" No chunks found in the input file. Nothing to embed.")
        return

    ensure_persist_dir(PERSIST_DIR)
    embed_and_store(ids, docs, metadatas)

if __name__ == "__main__":
    main()