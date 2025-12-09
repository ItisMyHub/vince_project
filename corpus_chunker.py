import json
import uuid

# --- CONFIGURATION ---
INPUT_FILE = "clean_corpus.json"
OUTPUT_FILE = "chunked_corpus.json"
CHUNK_SIZE = 1500  # Characters per chunk (approx 300-400 words)
OVERLAP = 200      # Characters of overlap to maintain context

def chunk_text(text, chunk_size, overlap):
    """Splits text into overlapping chunks."""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Move forward, but back up by 'overlap' amount
        start += (chunk_size - overlap)
    return chunks

def main():
    print(f"📖 Reading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            corpus = json.load(f)
    except FileNotFoundError:
        print("❌ File not found. Run data_cleaner.py first.")
        return

    chunked_data = []
    
    print("⚙️ Chunking data and preserving metadata...")
    
    for entry in corpus:
        content = entry.get("content", "")
        base_metadata = {
            "filename": entry.get("filename"),
            "filepath": entry.get("filepath"),
            "type": entry.get("type"),
            # PRESERVE THE NEW METADATA HERE
            "country": entry.get("country", "unknown"),
            "partner": entry.get("partner", "general"),
            "page": entry.get("page")
        }

        # Create chunks
        text_chunks = chunk_text(content, CHUNK_SIZE, OVERLAP)
        
        for i, text in enumerate(text_chunks):
            chunk_entry = {
                "id": str(uuid.uuid4()),
                "chunk_index": i,
                "content": text,
                **base_metadata  # Copy all metadata into the chunk
            }
            chunked_data.append(chunk_entry)

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunked_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Chunking complete. Created {len(chunked_data)} chunks.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()