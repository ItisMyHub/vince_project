def build_where_clause(keywords: list[dict]) -> dict:
    """
    keywords: list of {"Key": "...", "Value": "..."} from LocalAIRequest.keywords
    Applies only _model/_partner/_topic for local flow.
    """
    lookup = {k["Key"]: k["Value"] for k in keywords}
    if lookup.get("_model") != "_local":
        # For non-local, might bypass filtering
        return {}

    partner = lookup.get("_partner")
    topic = lookup.get("_topic")

    conditions = []
    if partner:
        conditions.append({"_partner": {"$eq": partner}})
    if topic:
        conditions. append({"_topics": {"$contains":  topic}})
    
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