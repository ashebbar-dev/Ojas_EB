# curated_chunker_from_csv.py

import os
import requests
import json
import csv
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import argparse
import time

# --- Configuration ---
load_dotenv()
APPROVED_CATEGORIES = ['Advice', 'Research', 'Information']

# --- Part 1: CSV URL Reader ---
def get_urls_from_csv(csv_filepath: str) -> list[str]:
    """
    Reads the first column from a CSV file to get a list of URLs.
    """
    print(f"[*] Reading URLs from: {csv_filepath}")
    urls_to_crawl = []
    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                if row:  # Ensure the row is not empty
                    url = row[0].strip()
                    if url.startswith('http'): # Basic validation
                        urls_to_crawl.append(url)
                    else:
                        print(f"    [!] Skipping invalid or header row: '{row[0]}'")

        print(f"[+] Discovered {len(urls_to_crawl)} URLs in the CSV file.")
        return urls_to_crawl
    except FileNotFoundError:
        print(f"[!] Error: The file '{csv_filepath}' was not found.")
        return []
    except Exception as e:
        print(f"[!] An error occurred while reading the CSV file: {e}")
        return []


# --- Part 2: Verification Function (No changes needed) ---
def verify_page_category(url: str, session: requests.Session) -> bool:
    print(f"    -> Verifying category for: {url}")
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        category_div = soup.find('div', class_='field--field-content-label')
        if category_div:
            category_text = category_div.get_text(strip=True)
            if category_text in APPROVED_CATEGORIES:
                print(f"    [✓] VERIFIED. Category is '{category_text}'.")
                return True
            else:
                print(f"    [x] SKIPPING. Category '{category_text}' is not in approved list.")
                return False
        else:
            print("    [x] SKIPPING. No category label found.")
            return False
    except requests.exceptions.RequestException as e:
        print(f"    [!] Error during verification request for {url}: {e}")
        return False

# --- Part 3: The Flexible Heading Chunking Logic (No changes needed) ---
def chunk_page_structurally(url: str, session: requests.Session) -> list[dict]:
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
                print(f"    [i] Success: Found content container using: {selector}")
                content_container = container
                for ad_box in content_container.find_all('div', class_='is-box-cta'):
                    ad_box.decompose()
                break
        
        if not content_container:
            print(f"    [!] All strategies failed. Skipping.")
            return []

        PRIMARY_HEADING_TAGS = ["h2", "h3"]
        UNWANTED_HEADINGS = {'page contents', 'in this article', 'table of contents', 'related content'}
        
        current_heading = "Introduction"
        current_chunk_content = []
        
        content_tags = content_container.find_all(PRIMARY_HEADING_TAGS + ["h4", "p", "li", "ul", "ol", "blockquote"])

        for tag in content_tags:
            if tag.name in PRIMARY_HEADING_TAGS:
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

# --- Part 4: Main Orchestration (Updated for CSV input) ---
def main():
    parser = argparse.ArgumentParser(description="Crawl, verify, and chunk content from a list of URLs in a CSV file.")
    parser.add_argument("csv_filepath", help="The path to the CSV file containing URLs in the first column.")
    parser.add_argument("-o", "--output", default="output.json", help="Output JSON file name (default: output.json).")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({'User-Agent': 'DementiaBot-Crawler/1.0'})

    urls_to_process = get_urls_from_csv(args.csv_filepath)
    if not urls_to_process:
        print("[!] No valid URLs found in the CSV file. Exiting.")
        return

    print(f"\n[*] Starting to verify and process {len(urls_to_process)} URLs from the CSV file.")
    all_chunks = []
    
    for i, url in enumerate(urls_to_process, 1):
        print(f"\n--- Processing URL {i}/{len(urls_to_process)} ---")
        if verify_page_category(url, session):
            time.sleep(1) # Be a polite crawler
            page_chunks = chunk_page_structurally(url, session)
            if page_chunks: all_chunks.extend(page_chunks)
    
    print(f"\n[*] Curation complete. Total chunks created: {len(all_chunks)}")
    print(f"[*] Saving all chunks to '{args.output}'...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"\n[***] Curated knowledge base saved to {args.output}. [***]")

if __name__ == "__main__":
    main()