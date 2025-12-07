import os
import json
import hashlib
from bs4 import BeautifulSoup
import fitz  # This is PyMuPDF

# --- CONFIGURATION ---
SOURCE_FOLDER = "./raw_data"  # Where your messy folders are
OUTPUT_FILE = "clean_corpus.json"
MIN_WORD_COUNT = 20

# --- HELPER FUNCTIONS ---

def get_file_hash(content):
    """Creates a unique fingerprint for text. If two files have the same hash, they are duplicates."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def read_html(filepath):
    """Extracts text from HTML, ignoring scripts and styles."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'html.parser')
        # Kill all script and style elements
        for script in soup(["script", "style", "nav", "footer"]):
            script.decompose()
        text = soup.get_text(separator=' ')
        return " ".join(text.split()) # Clean up extra whitespace

def read_pdf(filepath):
    """Extracts text from PDF."""
    text = ""
    try:
        doc = fitz.open(filepath)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
    return " ".join(text.split())

def read_md(filepath):
    """Reads Markdown (and treats it mostly like text)."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

# --- MAIN EXECUTION ---

def main():
    print("🧹 The Janitor is starting...")
    
    processed_data = []
    seen_hashes = set()
    
    # file_counter = 0

    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            content = ""
            file_type = ""

            # 1. Identify and Read
            if file.lower().endswith(".html"):
                content = read_html(file_path)
                file_type = "html"
            elif file.lower().endswith(".pdf"):
                content = read_pdf(file_path)
                file_type = "pdf"
            elif file.lower().endswith(".md"):
                content = read_md(file_path)
                file_type = "markdown"
            else:
                continue # Skip weird files

            # 2. Filter Garbage (Too short?)
            word_count = len(content.split())
            if word_count < MIN_WORD_COUNT:
                print(f"🗑️ Trash found ({word_count} words): {file}")
                continue

            # 3. Filter Duplicates (Already seen?)
            content_hash = get_file_hash(content)
            if content_hash in seen_hashes:
                print(f"👯 Duplicate found: {file}")
                continue
            
            # 4. Keep it!
            seen_hashes.add(content_hash)
            processed_data.append({
                "filename": file,
                "filepath": file_path,
                "language": "unknown", # You can add a language detector lib later if needed
                "type": file_type,
                "content": content
            })
            # file_counter += 1

    # 5. Save to JSON
    print(f"✨ Cleaning complete! Saved {len(processed_data)} documents.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()