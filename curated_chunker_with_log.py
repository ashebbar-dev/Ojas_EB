# curated_chunker.py

import os
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import argparse
import time

# --- Configuration ---
load_dotenv()
# List of categories we want to include in our knowledge base
APPROVED_CATEGORIES = ['Advice', 'Research', 'Information']
# --- NEW ---
# Define the acceptable range for the number of chunks from a single page.
MIN_CHUNKS = 4
MAX_CHUNKS = 7

# --- Part 1: Sitemap Parser ---
def get_urls_from_sitemap(sitemap_url: str, url_prefix: str) -> list[str]:
    print(f"[*] Fetching sitemap from: {sitemap_url}")
    matching_urls = []
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml-xml')
        loc_tags = soup.find_all('loc')
        
        for loc in loc_tags:
            url = loc.get_text().strip()
            if url.startswith(url_prefix):
                matching_urls.append(url)
        
        print(f"[+] Discovered {len(matching_urls)} URLs matching the prefix '{url_prefix}'.")
        return matching_urls

    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching sitemap: {e}")
        return []

# --- Part 2: The New Verification Function ---
def verify_page_category(url: str, session: requests.Session) -> bool:
    """
    Visits a page and checks if it belongs to one of the approved categories.
    """
    print(f"    -> Verifying category for: {url}")
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the specific div that contains the category label
        category_div = soup.find('div', class_='field--field-content-label')
        
        if category_div:
            category_text = category_div.get_text(strip=True)
            if category_text in APPROVED_CATEGORIES:
                print(f"    [✓] VERIFIED. Category is '{category_text}'.")
                return True
            else:
                print(f"    [x] SKIPPING. Category '{category_text}' is not in the approved list.")
                return False
        else:
            print("    [x] SKIPPING. No category label found on this page.")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"    [!] Error during verification request for {url}: {e}")
        return False

# --- Part 3: The Proven Chunking Logic ---
def chunk_page_structurally(url: str, session: requests.Session) -> list[dict]:
    # This is the same robust chunking logic from your final structural_chunker.py
    # ... (code is the same, just included here for completeness) ...
    print(f"    -> Chunking verified page: {url}")
    chunks = []
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        page_title = soup.title.string if soup.title else "No Title Found"

        content_container = None
        selectors_to_try = [
            {'name': 'div', 'class_': 'node--article--content'},
            {'name': 'article'},
            {'name': 'div', 'class_': 'column is-6'}
        ]

        for selector in selectors_to_try:
            container = soup.find(selector.get('name'), class_=selector.get('class_'))
            if container:
                content_container = container
                for ad_box in content_container.find_all('div', class_='is-box-cta'):
                    ad_box.decompose()
                break
        
        if not content_container: return []

        PRIMARY_HEADING_TAG = "h2"
        UNWANTED_HEADINGS = {'page contents', 'in this article', 'table of contents', 'related content'}
        current_heading = "Introduction"
        current_chunk_content = []
        content_tags = content_container.find_all([PRIMARY_HEADING_TAG, "h3", "h4", "p", "li", "ul", "ol", "blockquote"])

        for tag in content_tags:
            if tag.name == PRIMARY_HEADING_TAG:
                if current_chunk_content:
                    text = ' '.join(current_chunk_content).strip()
                    if text: chunks.append({"text": text,"metadata": {"source_url": url, "page_title": page_title, "topic_heading": current_heading}})
                new_heading_text = tag.get_text(separator=' ', strip=True)
                if new_heading_text.lower() in UNWANTED_HEADINGS:
                    current_heading = None; current_chunk_content = []
                else:
                    current_heading = new_heading_text; current_chunk_content = [current_heading]
            elif current_heading is not None:
                text = tag.get_text(separator=' ', strip=True)
                if text: current_chunk_content.append(text)
        
        if current_heading and current_chunk_content:
            text = ' '.join(current_chunk_content).strip()
            if text: chunks.append({"text": text, "metadata": {"source_url": url, "page_title": page_title, "topic_heading": current_heading}})

        print(f"    [+] Found and created {len(chunks)} chunks from this page.")
        return chunks
    except Exception as e:
        print(f"    [!] An unexpected error occurred while chunking {url}: {e}")
        return []

# --- Part 4: Main Orchestration (MODIFIED) ---
def main():
    parser = argparse.ArgumentParser(description="Discover, verify, and chunk content from a sitemap.")
    parser.add_argument("sitemap_url", help="The full URL of the sitemap.xml file.")
    parser.add_argument("url_prefix", help="The URL prefix to filter which pages to process (e.g., 'https://www.alzheimers.org.uk/blog').")
    parser.add_argument("-o", "--output", default="curated_knowledge_base.json", help="Output JSON file for the curated content.")
    # --- NEW ---
    parser.add_argument("-r", "--review-file", default="outliers_for_review.csv", help=f"Output file for URLs with chunk counts outside the range {MIN_CHUNKS}-{MAX_CHUNKS}.")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({'User-Agent': 'DementiaBot-Crawler/1.0'})

    urls_to_process = get_urls_from_sitemap(args.sitemap_url, args.url_prefix)
    if not urls_to_process:
        print("[!] No URLs discovered. Exiting.")
        return

    print(f"\n[*] Starting to verify and process {len(urls_to_process)} discovered URLs.")
    all_chunks = []
    
    # --- MODIFIED ---
    # Open the review file once and write to it as we go.
    # Using 'w' mode to create a new file each run, and newline='' for proper CSV handling.
    with open(args.review_file, 'w', encoding='utf-8', newline='') as review_file:
        # Write a header to the CSV file
        review_file.write("url,num_chunks\n")
        
        for i, url in enumerate(urls_to_process, 1):
            print(f"\n--- Processing URL {i}/{len(urls_to_process)} ---")
            
            if verify_page_category(url, session):
                time.sleep(1) 
                page_chunks = chunk_page_structurally(url, session)
                
                # --- NEW: Check the number of chunks and flag if it's an outlier ---
                num_chunks = len(page_chunks)
                # This condition also flags pages that failed to chunk (num_chunks = 0)
                if num_chunks < MIN_CHUNKS or num_chunks > MAX_CHUNKS:
                    print(f"    [!] FLAGGING FOR REVIEW. Found {num_chunks} chunks (outside {MIN_CHUNKS}-{MAX_CHUNKS} range).")
                    review_file.write(f"{url},{num_chunks}\n")

                if page_chunks:
                    all_chunks.extend(page_chunks)
    
    print(f"\n[*] Curation complete. Total chunks created: {len(all_chunks)}")
    print(f"[*] Saving all chunks to '{args.output}'...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    # --- MODIFIED ---
    print(f"\n[***] Curated knowledge base saved in '{args.output}'.")
    print(f"[***] URLs needing review have been saved to '{args.review_file}'. [***]")


if __name__ == "__main__":
    main()