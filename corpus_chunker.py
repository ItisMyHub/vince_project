import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
INPUT_FILE = "clean_corpus.json"
OUTPUT_FILE = "chunked_corpus.json"

# We are defining how large each piece of text should be.
# 512 characters is a safe balance for multilingual content on a CPU.
CHUNK_SIZE = 1500 

# We are defining the overlap.
# This ensures that if a sentence is cut in the middle, the context carries over to the next chunk.
CHUNK_OVERLAP = 200 

def main():
    # We are loading the clean data we prepared in the previous step.
    print(f" We are opening {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # We are initializing the splitter tool.
    # It will try to split on paragraphs first, then newlines, then spaces.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunked_data = []

    print(f"🔪 We are processing {len(data)} documents...")

    for document in data:
        original_text = document["content"]
        
        # We are actually splitting the text here.
        chunks = text_splitter.split_text(original_text)

        # We are creating a new entry for every single chunk.
        # We keep the metadata (filename, language) attached to each small piece.
        for i, chunk_text in enumerate(chunks):
            chunk_entry = {
                "id": f"{document['filename']}_chunk_{i}", # Unique ID
                "filename": document["filename"],
                "type": document["type"],
                "content": chunk_text, # The actual slice of text
                "original_source": document["filepath"]
            }
            chunked_data.append(chunk_entry)

    # We are saving the final results to a new JSON file.
    print(f" We are saving {len(chunked_data)} total chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=4)
        
    print("✅ Chunking complete. We are ready for the Embedding step.")

if __name__ == "__main__":
    main()