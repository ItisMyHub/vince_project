import chromadb

def build_where_clause(keywords: list[dict]) -> dict:
    """
    keywords: list of {"Key": "...", "Value": "..."} from LocalAIRequest.keywords
    Applies only _model/_partner/_topic for local flow.
    """
    lookup = {k["Key"]: k["Value"] for k in keywords}
    if lookup. get("_model") != "_local": 
        return {}

    partner = lookup.get("_partner")
    topic = lookup.get("_topic")

    conditions = []
    if partner:
        conditions.append({"_partner": {"$eq": partner}})
    if topic: 
        conditions.append({"_topics":  {"$contains": topic}})
    
    if not conditions: 
        where = {}
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}
    
    print(f"🔍 Filter -> Partner: {partner} | Topic: {topic} | Where: {where}")
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
    
    print("🧠 Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"✅ Connected!  Collection has {collection. count()} vectors.\n")
    
    # --- TEST 1: TUAS + Residence Permit ---
    print("=" * 60)
    print("❓ Question: How do I apply for a residence permit?\n")
    keywords = [
        {"Key": "_model", "Value": "_local"},
        {"Key": "_partner", "Value": "_tuas"},
        {"Key": "_topic", "Value": "_residencepermit"}
    ]
    
    results = retrieve(collection, "How do I apply for a residence permit? ", top_k=3, keywords=keywords)
    
    print(f"\n✅ Found {len(results['documents'][0])} results:\n")
    for i, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][i]
        print(f"--- Result {i+1} ---")
        print(f"📁 File: {meta. get('filename')}")
        print(f"🤝 Partner: {meta. get('_partner')}")
        print(f"🏷️  Topics: {meta. get('_topics')[: 60]}...")
        print(f"📄 Content:  {doc[: 150]}...\n")