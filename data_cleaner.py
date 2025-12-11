import os
import json
import hashlib
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
SOURCE_FOLDER = "./raw_data"
OUTPUT_FILE = "clean_corpus.json"
MIN_WORD_COUNT = 20  # Skip very short files

# Default tags so every chunk has the keys the client will filter on
DEFAULT_MODEL = "_local"
DEFAULT_COUNTRY = "unknown"
DEFAULT_REGION = "unknown"
DEFAULT_CITY = "unknown"
DEFAULT_PARTNER = "general"

def get_file_hash(content):
    """Generates an MD5 hash of the text content to detect duplicates."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def clean_html(file_path):
    """Extracts text from HTML, removing scripts/styles/headers."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return [{"text": text, "page": None}]
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
            text = " ".join(text.split())
            if len(text) > 50:
                pages_data.append({
                    "text": text,
                    "page": page_num + 1
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
    Infers country/region/city/partner from the folder structure.
    """
    norm_path = os.path.normpath(file_path)
    parts = [p.lower() for p in norm_path.split(os.sep)]

    country = DEFAULT_COUNTRY
    region = DEFAULT_REGION
    city = DEFAULT_CITY
    partner = DEFAULT_PARTNER

    try:
        idx = parts.index("raw_data")
        tail = parts[idx + 1 : -1]  # segments after raw_data, before filename

        if len(tail) >= 1:
            country = tail[0]
        if len(tail) >= 2:
            region = tail[1]
        if len(tail) >= 3:
            city = tail[2]
        if tail:
            partner = tail[-1]

        # General Info override: countrywide scope
        if "general_info" in tail or "general info" in tail:
            region = "countrywide"
            city = "countrywide"
            if partner == country:
                partner = DEFAULT_PARTNER

        # Avoid partner == country
        if partner == country:
            partner = DEFAULT_PARTNER

    except ValueError:
        pass

    return country, region, city, partner

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

            extracted_pages = []
            if ext == "html":
                extracted_pages = clean_html(file_path)
            elif ext == "pdf":
                extracted_pages = clean_pdf(file_path)
            elif ext in ["md", "txt"]:
                extracted_pages = clean_markdown(file_path)
            else:
                continue

            country, region, city, partner = extract_metadata_from_path(file_path)

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
                    "region": region,
                    "city": city,
                    "partner": partner,
                    "_model": DEFAULT_MODEL,
                    "_country": country,
                    "_region": region,
                    "_city": city,
                    "_partner": partner,
                    "page": page_num,
                    "content": text_content
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)

    print(f" Cleaned corpus saved to {OUTPUT_FILE} with {len(corpus)} entries.")
    if corpus:
        example = corpus[-1]
        print(f"Example Check -> File: {example['filename']} | Country: {example['country']} | Partner: {example['partner']} | Region: {example['region']} | City: {example['city']}")

if __name__ == "__main__":
    main()