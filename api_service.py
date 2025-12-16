import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests
import time

# --- CONFIGURATION ---
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "vince_agent"
EMBEDDING_MODEL = "BAAI/bge-m3"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# --- INITIALIZE ---
print("🧠 Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("🔗 Connecting to ChromaDB...")
client = chromadb. PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"✅ Connected!  {collection.count()} vectors loaded.")

# --- FASTAPI APP ---
app = FastAPI(title="Vince Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REQUEST MODELS (matching Unity client) ---
class Keyword(BaseModel):
    Key: str
    Value: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    keywords: list[Keyword] = []

# --- RESPONSE MODELS (matching mobile app expectations) ---
class SourceItem(BaseModel):
    file: str
    page: int | None
    section: str | None
    similarity:  float
    original_indices: list[int]

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    retrieval_count: int

# --- HELPER FUNCTIONS ---
def build_where_clause(keywords:  list[Keyword]) -> dict:
    """Build Chroma where clause from keywords."""
    lookup = {k.Key: k.Value for k in keywords}
    
    if lookup.get("_model") != "_local":
        return {}
    
    partner = lookup.get("_partner")
    
    where = {}
    if partner:
        where["_partner"] = {"$eq": partner}
    
    print(f"🔍 Chroma Filter -> Partner: {partner}")
    return where

def retrieve(question: str, keywords: list[Keyword], top_k:  int):
    """Retrieve relevant chunks from vector store."""
    lookup = {k.Key: k.Value for k in keywords}
    topic = lookup.get("_topic")
    
    where = build_where_clause(keywords)
    
    # Embed question
    query_embedding = embedder. encode(question, normalize_embeddings=True).tolist()
    
    # Query Chroma (fetch extra for post-filtering)
    fetch_k = top_k * 3 if topic else top_k
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        where=where if where else None,
        include=["documents", "metadatas", "distances"]
    )
    
    # Post-filter by topic if specified
    docs = []
    metas = []
    distances = []
    
    for i, meta in enumerate(results['metadatas'][0]):
        if topic: 
            topics_str = meta. get('_topics', '[]')
            if topic not in topics_str: 
                continue
        
        docs.append(results['documents'][0][i])
        metas.append(meta)
        distances.append(results['distances'][0][i])
        
        if len(docs) >= top_k:
            break
    
    if topic:
        print(f"🏷️ Topic Filter -> '{topic}' | Matched:  {len(docs)}")
    
    return docs, metas, distances

def generate_answer(question: str, docs: list, metas: list) -> str:
    """Generate answer using Ollama LLM."""
    if not docs:
        return "I couldn't find relevant information to answer your question."
    
    # Build context with source markers
    context_parts = []
    for i, doc in enumerate(docs):
        context_parts. append(f"[SOURCE_{i+1}]: {doc}")
    context = "\n\n".join(context_parts)
    
    prompt = f"""Answer the question using ONLY information from the provided SOURCES.
    Cite as [SOURCE_1], [SOURCE_2] in your answer when you use information from that source.
    The ansewer must be pulled from the cited sources. 
   
 SOURCES:
 {context}

 QUESTION: {question}

 ANSWER:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "60m"
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json().get("response", "Error generating response.")
    except requests. exceptions.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}. Please ensure Ollama is running."

def convert_distance_to_similarity(distance: float) -> float:
    """Convert Chroma distance to similarity score (0-1)."""
    # Chroma returns L2 distance for normalized embeddings
    # Similarity = 1 - (distance / 2) for normalized vectors
    similarity = 1 - (distance / 2)
    return round(max(0, min(1, similarity)), 6)

def build_sources(docs: list, metas: list, distances:  list) -> list[SourceItem]:
    """Build sources array matching the expected schema."""
    sources = []
    
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        # Get page (convert to int if exists)
        page_raw = meta.get("page")
        page = None
        if page_raw and page_raw != "0" and page_raw != "None":
            try:
                page = int(page_raw)
            except (ValueError, TypeError):
                page = None
        
        # Get section (for HTML files)
        section = meta.get("section", None)
        
        # Convert distance to similarity
        similarity = convert_distance_to_similarity(dist)
        
        sources.append(SourceItem(
            file=meta.get("filename", "unknown"),
            page=page,
            section=section,
            similarity=similarity,
            original_indices=[i + 1]
        ))
    
    return sources

# --- ENDPOINTS ---
@app. get("/")
def root():
    return {"status": "ok", "message": "Vince Agent API is running"}

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "vectors":  collection.count(),
        "ollama_model": OLLAMA_MODEL
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Main endpoint matching Unity client POST /query schema."""
    start_time = time. time()
    
    print(f"\n{'='*60}")
    print(f"❓ Question: {request.question}")
    print(f"📊 Top K: {request.top_k}")
    print(f"🔑 Keywords: {[(k.Key, k.Value) for k in request.keywords]}")
    
    # Retrieve relevant chunks
    docs, metas, distances = retrieve(
        question=request. question,
        keywords=request.keywords,
        top_k=request.top_k
    )
    
    print(f"📄 Retrieved {len(docs)} chunks")
    
    # Generate answer with LLM
    answer = generate_answer(request.question, docs, metas)
    
    # Build sources array
    sources = build_sources(docs, metas, distances)
    
    processing_time = time. time() - start_time
    print(f"✅ Response generated in {processing_time:.2f}s")
    
    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        retrieval_count=len(sources)
    )

# --- RUN ---
if __name__ == "__main__": 
    import uvicorn
    print(f"\n🚀 Starting Vince Agent API on http://localhost:8000")
    print(f"📖 API Docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)