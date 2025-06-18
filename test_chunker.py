# test_chunker.py

import requests
import json
from bs4 import BeautifulSoup
import argparse

# --- The Core Chunking Logic ---
# This is the final, most robust version of the chunking function.
def chunk_page_structurally(url: str, session: requests.Session) -> list[dict]:
    """
    Crawls and chunks a single web page based on its HTML structure.
    This function contains the final, prioritized logic for finding content.
    """
    print(f"    -> Attempting to chunk: {url}")
    chunks = []
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        page_title = soup.title.string.strip() if soup.title else "No Title Found"

        # --- The Prioritized Selector Logic ---
        content_container = None
        selectors_to_try = [
            {'name': 'div', 'class_': 'node--article--content'}, # Priority 1: Specific blog post container
            {'name': 'article'},                              # Priority 2: General article container
            {'name': 'div', 'class_': 'column is-6'}           # Priority 3: Fallback layout container
        ]

        for selector in selectors_to_try:
            container = soup.find(selector.get('name'), class_=selector.get('class_'))
            if container:
                print(f"    [i] Success: Found content container using selector: {selector}")
                content_container = container
                # Actively remove known "noise" elements before chunking
                for ad_box in content_container.find_all('div', class_='is-box-cta'):
                    ad_box.decompose()
                break # Stop searching once the highest-priority container is found
        
        if not content_container:
            print(f"    [!] FAILED: Could not find a suitable content container on this page.")
            return []
        # --- End of Selector Logic ---

        # Flexible heading tags to split on
        PRIMARY_HEADING_TAGS = ["h2", "h3"]
        UNWANTED_HEADINGS = {'page contents', 'in this article', 'table of contents', 'related content'}

        current_heading = "Introduction"
        current_chunk_content = []
        
        content_tags = content_container.find_all(PRIMARY_HEADING_TAGS + ["h4", "p", "li", "ul", "ol", "blockquote"])

        for tag in content_tags:
            if tag.name in PRIMARY_HEADING_TAGS:
                if current_chunk_content:
                    text = ' '.join(current_chunk_content).strip()
                    if text: chunks.append({"text": text, "metadata": {"source_url": url, "page_title": page_title, "topic_heading": current_heading}})
                
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
        print(f"    [!] An unexpected error occurred while processing {url}: {e}")
        return []

# --- Main Orchestration for a Single URL ---
def main():
    parser = argparse.ArgumentParser(description="Test the structural chunking logic on a single URL.")
    parser.add_argument("url", help="The full URL of the single page you want to test.")
    parser.add_argument("-o", "--output", default="test_output.json", help="The name of the output JSON file (default: test_output.json).")
    args = parser.parse_args()

    # Use a requests.Session for good practice
    session = requests.Session()
    session.headers.update({'User-Agent': 'DementiaBot-Crawler-Tester/1.0'})

    print(f"\n[*] Starting test on a single URL: {args.url}")
    
    # Run the chunker on the single provided URL
    page_chunks = chunk_page_structurally(args.url, session)
    
    if not page_chunks:
        print("\n[***] Test complete. No chunks were created. [***]")
        return
        
    print(f"\n[*] Test complete. Total chunks created: {len(page_chunks)}")
    print(f"[*] Saving chunked output to '{args.output}'...")
    
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(page_chunks, f, indent=2, ensure_ascii=False)
        print(f"\n[***] All done! The test output is saved in '{args.output}'. [***]")
    except IOError as e:
        print(f"[!] Error writing to file '{args.output}': {e}")


if __name__ == "__main__":
    main()