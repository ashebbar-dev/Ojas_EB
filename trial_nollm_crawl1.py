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
    """
    Fetches a sitemap, parses it, and returns a list of URLs
    that start with the specified prefix.
    """
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

# --- Part 2: Structural Chunker (Revised and Improved) ---
def chunk_page_structurally(url: str) -> list[dict]:
    """
    Crawls a single URL, finds the <article> tag, and chunks its content
    based on a primary heading tag (h2). Sub-headings (h3, h4) are included
    within the parent chunk.

    Args:
        url: The URL of the page to crawl and chunk.

    Returns:
        A list of chunk dictionaries.
    """
    print(f"    -> Chunking: {url}")
    chunks = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        page_title = soup.title.string if soup.title else "No Title Found"
        article_body = soup.find('article')
        
        if not article_body:
            print(f"    [!] Warning: No <article> tag found on {url}. Skipping.")
            return []

        # --- CONFIGURATION FOR CHUNKING LOGIC ---
        PRIMARY_HEADING_TAG = "h2"
        # Common navigational/meta headings to ignore. Add to this list as needed.
        UNWANTED_HEADINGS = {'page contents', 'in this article', 'table of contents', 'related content'}

        current_heading = "Introduction"
        current_chunk_content = []

        # Find all relevant content tags in order
        content_tags = article_body.find_all([PRIMARY_HEADING_TAG, "h3", "h4", "p", "li", "ul", "ol"])

        for tag in content_tags:
            # FIX 3: A new chunk is ONLY started by the PRIMARY_HEADING_TAG
            if tag.name == PRIMARY_HEADING_TAG:
                # First, save the previous chunk being built
                if current_chunk_content:
                    # FIX 1: Join with spaces to prevent words sticking together
                    text = ' '.join(current_chunk_content).strip()
                    if text:
                        chunks.append({
                            "text": text,
                            "metadata": {
                                "source_url": url,
                                "page_title": page_title,
                                "topic_heading": current_heading
                            }
                        })
                
                # Now, start the new chunk
                new_heading_text = tag.get_text(separator=' ', strip=True)
                
                # FIX 2: Check if the new heading is in our blocklist
                if new_heading_text.lower() in UNWANTED_HEADINGS:
                    current_heading = None # Temporarily disable adding content
                    current_chunk_content = []
                else:
                    current_heading = new_heading_text
                    # The heading text itself is part of the chunk
                    current_chunk_content = [current_heading]
            
            # This handles all other tags (h3, h4, p, li)
            elif current_heading is not None:
                # FIX 1: Use separator=' ' to ensure proper spacing
                text = tag.get_text(separator=' ', strip=True)
                if text:
                    # Append the content of subheadings and paragraphs to the current chunk
                    current_chunk_content.append(text)
        
        # After the loop, save the final chunk
        if current_heading and current_chunk_content:
            text = ' '.join(current_chunk_content).strip()
            if text:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "source_url": url,
                        "page_title": page_title,
                        "topic_heading": current_heading
                    }
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
    """Main function to run the crawler and chunker."""
    parser = argparse.ArgumentParser(
        description="Crawl URLs from a sitemap, perform structural chunking, and save to a JSON file."
    )
    parser.add_argument("sitemap_url", help="The full URL of the sitemap.xml file.")
    parser.add_argument("url_prefix", help="The URL prefix to filter which pages to crawl.")
    parser.add_argument(
        "-o", "--output", 
        default="knowledge_base.json", 
        help="The name of the output JSON file (default: knowledge_base.json)."
    )
    
    args = parser.parse_args()
    urls_to_process = get_urls_from_sitemap(args.sitemap_url, args.url_prefix)
    if not urls_to_process:
        print("[!] No URLs found matching the prefix. Exiting.")
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