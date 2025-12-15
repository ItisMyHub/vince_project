import os
import requests
from urllib.parse import urlparse

# --- CONFIGURATION ---
# Listing the specific URLs to teach the agent.
TARGET_URLS = [
    "https://migri.fi/en/home/",
    "https://www.infofinland.fi/",
    "https://www.turkuamk.fi/"
    # We can add more URLs here...
]

OUTPUT_FOLDER = "./raw_data"

def main():
    # Ensuring the output folder exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f" Started to scrape {len(TARGET_URLS)} pages...")

    for url in TARGET_URLS:
        try:
            print(f" Downloading: {url}")
            # Fetching the page content
            response = requests.get(url)
            response.raise_for_status() # Check for errors (like 404 Not Found)

            # Creating a valid filename from the URL
            # e.g. "fastapi.tiangolo.com/tutorial/" -> "fastapi_tiangolo_com_tutorial_.html"
            parsed_url = urlparse(url)
            filename = parsed_url.netloc + parsed_url.path
            filename = filename.replace("/", "_").replace(".", "_").strip("_") + ".html"
            
            save_path = os.path.join(OUTPUT_FOLDER, filename)

            # Saving the raw HTML directly to the raw_data folder
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            print(f" Saved to: {save_path}")

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    print(" Scraping complete. Now run data_cleaner.py ")

if __name__ == "__main__":
    main()