import chromadb
from sentence_transformers import SentenceTransformer

def build_where_clause(keywords: list[dict]) -> dict:
    lookup = {k["Key"]: k["Value"] for k in keywords}
    if lookup.get("_model") != "_local":
        return {}

    partner = lookup.get("_partner")
    topic_val = lookup.get("_topic")

    conditions = []
    if partner:
        conditions.append({"_partner": {"$eq": partner}})
    if topic_val:
        # Ensure list for $in; if topic_val is a string, wrap it
        topic_list = topic_val if isinstance(topic_val, list) else [topic_val]
        conditions.append({"_topics": {"$in": topic_list}})

    if not conditions:
        where = {}
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    print(f"🔍 Filter -> Partner: {partner} | Topic: {topic_val} | Where: {where}")
    return where

def retrieve(collection, question: str, top_k: int, keywords: list[dict]):
    where = build_where_clause(keywords)
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where if where else None,
    )
    return results

if __name__ == "__main__":
    PERSIST_DIR = "./chroma_db"
    COLLECTION_NAME = "vince_agent"
    MODEL_NAME = "BAAI/bge-m3"
    
    print("🧠 Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    
    print("🔗 Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"✅ Connected! Collection has {collection.count()} vectors.\n")
    
    # Embed the question with the SAME model
    question = "How do I apply for a residence permit?"
    query_embedding = model.encode(question, normalize_embeddings=True).tolist()
    
    # Query with embedding instead of text
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"_partner": {"$eq": "_tuas"}}
    )
    
    print(f"✅ Found {len(results['documents'][0])} results:\n")
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        print(f"--- Result {i+1} ---")
        print(f"📁 File: {meta. get('filename')}")
        print(f"📄 Content: {doc[: 150]}...\n")