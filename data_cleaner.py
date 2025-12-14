import os
import json
import hashlib
import fitz  # PyMuPDF
from pathlib import Path
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
SOURCE_FOLDER = "./raw_data"
OUTPUT_FILE = "clean_corpus.json"
MIN_WORD_COUNT = 20  # Skip very short files

# Defaults (legacy fields kept for compatibility; ignored in local filters)
DEFAULT_MODEL = "_local"
DEFAULT_COUNTRY = "unknown"
DEFAULT_REGION = "unknown"
DEFAULT_CITY = "unknown"
DEFAULT_PARTNER = "general"

def get_file_hash(content: str) -> str:
    """Generates an MD5 hash of the text content to detect duplicates."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()

def clean_html(file_path: str):
    """Extracts text from HTML, removing scripts/styles/nav/footer/header."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        return [{"text": text, "page": None}]
    except Exception as e:
        print(f" Error reading HTML {file_path}: {e}")
        return []

def clean_pdf(file_path: str):
    """Extracts text from PDF, keeping page numbers."""
    pages_data = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            text = " ".join(text.split())
            if len(text) > 50:
                pages_data.append({"text": text, "page": page_num + 1})
        doc.close()
    except Exception as e:
        print(f" Error reading PDF {file_path}: {e}")
    return pages_data

def clean_markdown(file_path: str):
    """Reads Markdown/txt files."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{"text": text, "page": None}]
    except Exception as e:
        print(f" Error reading MD {file_path}: {e}")
        return []

def infer_partner(path: Path) -> str:
    """Infers partner from path segments: TUAS, Sateenkaarikoto, FMC_CABO."""
    parts = [p.lower() for p in path.parts]
    if "tuas" in parts:
        return "_tuas"
    if "sateenkaarikoto" in parts:
        return "_sateenkaarikoto"
    if "fmc_cabo" in parts:
        return "_fmc_cabo"
    return DEFAULT_PARTNER

def load_topics_for_domain(domain_path: Path) -> list[str]:
    """Reads keywords.json in the domain folder; returns list of topic strings."""
    keywords_file = domain_path / "keywords.json"
    if not keywords_file.exists():
        print(f" Warning: no keywords.json in {domain_path}")
        return []
    try:
        with keywords_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        topics = [t for t in data if isinstance(t, str)]
        return topics
    except Exception as e:
        print(f" Error reading keywords.json in {domain_path}: {e}")
        return []

def extract_metadata_from_path(file_path: str):
    """
    Infers country/region/city/partner from the folder structure.
    Legacy fields are retained for compatibility but ignored in local retrieval.
    Partner is inferred explicitly.
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

        # Explicit partner inference
        partner = infer_partner(Path(file_path))

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
            ext = filename.lower().split(".")[-1]

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

            # Domain folder assumed to be the parent; adjust if deeper nesting is used
            domain_path = Path(file_path).parent
            topics = load_topics_for_domain(domain_path)

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
                    "country": country,   # legacy
                    "region": region,     # legacy
                    "city": city,         # legacy
                    "partner": partner,   # legacy-friendly
                    "_model": DEFAULT_MODEL,
                    "_country": country,  # legacy
                    "_region": region,    # legacy
                    "_city": city,        # legacy
                    "_partner": partner,  # used for filtering
                    "_topics": topics,    # used for filtering ($contains)
                    "page": page_num,
                    "content": text_content
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)

    print(f" Cleaned corpus saved to {OUTPUT_FILE} with {len(corpus)} entries.")
    if corpus:
        example = corpus[-1]
        print(f"Example Check -> File: {example['filename']} | Partner: {example['_partner']} | Topics: {example['_topics']}")

if __name__ == "__main__":
    main()