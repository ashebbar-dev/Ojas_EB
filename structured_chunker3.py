# structural_chunker.py (Final Definitive Version)

import os
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import argparse

# --- Configuration ---
load_dotenv()

# --- Part 1: Sitemap Parser (Unchanged) ---
def get_urls_from_sitemap(sitemap_url: str, url_prefix: str) -> list[str]:
    # ... (code is the same, omitted for brevity) ...
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
        
        print(f"[+] Found {len(matching_urls)} URLs matching the prefix '{url_prefix}'.")
        return matching_urls

    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching sitemap: {e}")
        return []
    except Exception as e:
        print(f"[!] An unexpected error occurred while parsing the sitemap: {e}")
        return []

# --- Part 2: Structural Chunker (Final Prioritized Logic) ---
def chunk_page_structurally(url: str) -> list[dict]:
    print(f"    -> Chunking: {url}")
    chunks = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        page_title = soup.title.string if soup.title else "No Title Found"

        # --- THE NEW, CORRECTLY PRIORITIZED LOGIC ---
        content_container = None
        
        # This list defines our search priority. We look for the most specific
        # and reliable containers first.
        selectors_to_try = [
            # Priority 1: The "Golden Parent" for blog posts. It's very specific.
            {'name': 'div', 'class_': 'node--article--content'},
            
            # Priority 2: The container for main informational pages. More generic than the one above.
            {'name': 'article'},
            
            # Priority 3: A final fallback for other layouts. This is the least reliable.
            {'name': 'div', 'class_': 'column is-6'}
        ]

        for selector in selectors_to_try:
            container = soup.find(selector.get('name'), class_=selector.get('class_'))
            if container:
                print(f"    [i] Success: Found content container using priority selector: {selector}")
                content_container = container
                
                # Actively remove known "noise" like call-to-action boxes if they exist
                for ad_box in content_container.find_all('div', class_='is-box-cta'):
                    print("    [i] Removing call-to-action box from content.")
                    ad_box.decompose()
                
                break # Stop searching as soon as we find the highest-priority match
        
        if not content_container:
            print(f"    [!] Warning: All strategies failed. Could not find a suitable content container on {url}. Skipping.")
            return []
        # --- END OF NEW LOGIC ---

        PRIMARY_HEADING_TAG = "h2"
        UNWANTED_HEADINGS = {'page contents', 'in this article', 'table of contents', 'related content'}

        current_heading = "Introduction"
        current_chunk_content = []

        content_tags = content_container.find_all([PRIMARY_HEADING_TAG, "h3", "h4", "p", "li", "ul", "ol", "blockquote"])

        for tag in content_tags:
            # ... (the rest of the chunking logic is the same, omitted for brevity) ...
            if tag.name == PRIMARY_HEADING_TAG:
                if current_chunk_content:
                    text = ' '.join(current_chunk_content).strip()
                    if text:
                        chunks.append({
                            "text": text,
                            "metadata": {"source_url": url, "page_title": page_title, "topic_heading": current_heading}
                        })
                
                new_heading_text = tag.get_text(separator=' ', strip=True)
                
                if new_heading_text.lower() in UNWANTED_HEADINGS:
                    current_heading = None
                    current_chunk_content = []
                else:
                    current_heading = new_heading_text
                    current_chunk_content = [current_heading]
            
            elif current_heading is not None:
                text = tag.get_text(separator=' ', strip=True)
                if text:
                    current_chunk_content.append(text)
        
        if current_heading and current_chunk_content:
            text = ' '.join(current_chunk_content).strip()
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {"source_url": url, "page_title": page_title, "topic_heading": current_heading}
                })

        print(f"    [+] Found and created {len(chunks)} chunks from this page.")
        return chunks

    except requests.exceptions.RequestException as e:
        print(f"    [!] Error during request for {url}: {e}")
        return []
    except Exception as e:
        print(f"    [!] An unexpected error occurred while chunking {url}: {e}")
        return []

# --- Part 3: Main Orchestration (Unchanged) ---
def main():
    # ... (code is the same, omitted for brevity) ...
    parser = argparse.ArgumentParser(description="Crawl URLs, perform structural chunking, and save to a JSON file.")
    parser.add_argument("sitemap_url", help="The full URL of the sitemap.xml file.")
    parser.add_argument("url_prefix", help="The URL prefix to filter which pages to crawl.")
    parser.add_argument("-o", "--output", default="knowledge_base.json", help="Output JSON file.")
    args = parser.parse_args()
    urls_to_process = get_urls_from_sitemap(args.sitemap_url, args.url_prefix)
    if not urls_to_process:
        print("[!] No URLs found. Exiting.")
        return

    print(f"[*] Starting to process {len(urls_to_process)} URLs.")
    all_chunks = []
    
    for i, url in enumerate(urls_to_process, 1):
        print(f"\n--- Processing URL {i}/{len(urls_to_process)} ---")
        page_chunks = chunk_page_structurally(url)
        if page_chunks:
            all_chunks.extend(page_chunks)
    
    print(f"\n[*] Total chunks created: {len(all_chunks)}")
    print(f"[*] Saving all chunks to '{args.output}'...")
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"\n[***] All done! The chunked knowledge base is saved in '{args.output}'. [***]")
    except IOError as e:
        print(f"[!] Error writing to file '{args.output}': {e}")


if __name__ == "__main__":
    main()