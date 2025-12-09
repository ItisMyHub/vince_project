import os
import json
import hashlib
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
SOURCE_FOLDER = "./raw_data"
OUTPUT_FILE = "clean_corpus.json"
MIN_WORD_COUNT = 20  # Skip very short files

def get_file_hash(content):
    """Generates an MD5 hash of the text content to detect duplicates."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def clean_html(file_path):
    """Extracts text from HTML, removing scripts/styles/headers."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        
        # We are removing javascript, css, navigation, footer, header
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        
        # Basic cleanup of extra whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return [{"text": text, "page": None}] # HTML has no pages
        
    except Exception as e:
        print(f" Error reading HTML {file_path}: {e}")
        return []

def clean_pdf(file_path):
    """Extracts text from PDF, keeping track of page numbers."""
    pages_data = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            # clean up whitespace
            text = " ".join(text.split())
            if len(text) > 50:  # Skip mostly empty pages
                pages_data.append({
                    "text": text,
                    "page": page_num + 1  # Humans count from 1
                })
        doc.close()
    except Exception as e:
        print(f" Error reading PDF {file_path}: {e}")
    return pages_data

def clean_markdown(file_path):
    """Reads Markdown files."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{"text": text, "page": None}]
    except Exception as e:
        print(f" Error reading MD {file_path}: {e}")
        return []

def extract_metadata_from_path(file_path):
    """
    Infers 'Country' and 'Partner' from the folder structure.
    Logic:
    - Country = The folder directly inside 'raw_data' (e.g., Finland)
    - Partner = The folder directly containing the file (e.g., TUAS)
    """
    norm_path = os.path.normpath(file_path)
    parts = norm_path.split(os.sep)
    
    country = "unknown"
    partner = "general"

    try:
        if "raw_data" in parts:
            idx = parts.index("raw_data")
            
            # 1. Country is the first folder after raw_data
            if len(parts) > idx + 1:
                country = parts[idx + 1].lower()
            
            # 2. Partner is the immediate parent folder of the file
            # content is at parts[-1], so parent is parts[-2]
            if len(parts) > 1:
                partner = parts[-2].lower()
                
            # Edge case: if the file is directly inside raw_data/Finland
            if partner == country:
                partner = "general"

    except Exception:
        pass
        
    return country, partner

def main():
    if not os.path.exists(SOURCE_FOLDER):
        print(f" Source folder '{SOURCE_FOLDER}' not found.")
        return

    corpus = []
    seen_hashes = set()
    
    print(" Starting cleanup and metadata extraction...")

    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = filename.lower().split('.')[-1]
            
            # 1. Determine Extraction Method
            extracted_pages = []
            if ext == "html":
                extracted_pages = clean_html(file_path)
            elif ext == "pdf":
                extracted_pages = clean_pdf(file_path)
            elif ext in ["md", "txt"]:
                extracted_pages = clean_markdown(file_path)
            else:
                continue 

            # 2. Extract Metadata from Folder Structure
            country, partner = extract_metadata_from_path(file_path)

            # 3. Process content
            for item in extracted_pages:
                text_content = item["text"]
                page_num = item["page"]

                if len(text_content.split()) < MIN_WORD_COUNT:
                    continue

                content_hash = get_file_hash(text_content)
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                corpus.append({
                    "filename": filename,
                    "filepath": file_path,
                    "type": ext,
                    "country": country,
                    "partner": partner,
                    "page": page_num,
                    "content": text_content
                })

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)
    
    print(f" Cleaned corpus saved to {OUTPUT_FILE} with {len(corpus)} entries.")
    if corpus:
        # Show example to verify your tags
        example = corpus[-1]
        print(f"Example Check -> File: {example['filename']} | Country: {example['country']} | Partner: {example['partner']}")

if __name__ == "__main__":
    main()