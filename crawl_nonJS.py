import os
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import argparse

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()

# --- Part 1: Sitemap Parser ---
def get_urls_from_sitemap(sitemap_url: str, url_prefix: str) -> list[str]:
    """
    Fetches a sitemap, parses it, and returns a list of URLs
    that start with the specified prefix.

    Args:
        sitemap_url: The full URL of the sitemap.xml file.
        url_prefix: The prefix to filter URLs (e.g., 'https://example.com/blog/').

    Returns:
        A list of matching URLs.
    """
    print(f"[*] Fetching sitemap from: {sitemap_url}")
    matching_urls = []
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # We use the 'lxml-xml' or 'xml' parser for sitemaps
        soup = BeautifulSoup(response.content, 'lxml-xml')
        
        # Sitemaps use the <loc> tag to store the URL
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

# --- Part 2: Web Page Crawler ---
def crawl_page_content(url: str) -> dict | None:
    """
    Crawls a single URL and extracts its content based on the provided logic.
    This function is based on the code you provided.

    Args:
        url: The URL of the page to crawl.

    Returns:
        A dictionary containing the URL, title, and cleaned content, or None if failed.
    """
    print(f"    -> Crawling: {url}")
    try:
        response = requests.get(url, timeout=10)
        
        # Check if the request was successful
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main content area (using <article> as specified)
            article_body = soup.find('article')
            
            if article_body:
                # Get clean text from the main content
                clean_text = article_body.get_text(separator=' ', strip=True)
                
                # This is your data for the knowledge base
                knowledge_item = {
                    "url": url,
                    "title": soup.title.string if soup.title else "No Title Found",
                    "content": clean_text
                }
                return knowledge_item
            else:
                print(f"    [!] Warning: No <article> tag found on {url}. Skipping.")
                return None
        else:
            print(f"    [!] Warning: Failed to fetch {url}. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"    [!] Error during request for {url}: {e}")
        return None

# --- Part 3: LLM Interaction ---
def get_structured_content_from_llm(knowledge_item: dict, service: str) -> str | None:
    """
    Sends the crawled content to an LLM (OpenAI or OpenRouter) and asks it
    to structure the content.

    Args:
        knowledge_item: The dictionary from the crawl_page_content function.
        service: The LLM service to use ('openai' or 'openrouter').

    Returns:
        The structured content from the LLM, or None if an error occurs.
    """
    print(f"    -> Sending content from {knowledge_item['url']} to LLM ({service})...")

    # ---- Client Configuration (Handles both OpenAI and OpenRouter) ----
    if service == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[!] Error: OPENAI_API_KEY not found in .env file.")
            return None
        client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini" # Or "gpt-4-turbo" etc.
    
    elif service == 'openrouter':
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("[!] Error: OPENROUTER_API_KEY not found in .env file.")
            return None
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        # You can use models from different providers via OpenRouter
        model = "openai/gpt-3.5-turbo" # Example model
    else:
        print(f"[!] Error: Invalid LLM service '{service}'. Choose 'openai' or 'openrouter'.")
        return None
    # --------------------------------------------------------------------

    # The prompt for the LLM
    prompt = f"""
    You are an expert content structurer. Your task is to re-format the following raw web page content into a clean, well-structured document.

    **Instructions:**
    1.  Start the document with a clear title based on the provided content. Use a markdown H1 header (e.g., '# Title').
    2.  Structure the rest of the content logically using markdown for subheadings (##), lists (* or 1.), and bold text (**).
    3.  Ensure the final output is only the structured markdown content. Do not add any extra commentary, introductions, or conclusions like "Here is the structured content:".

    **Original Page Data:**
    - URL: {knowledge_item['url']}
    - Original Title: {knowledge_item['title']}

    **Raw Content to Structure:**
    ---
    {knowledge_item['content']}
    ---
    """

    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a content structuring assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5, # A little creativity but mostly factual
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"    [!] Error calling LLM API: {e}")
        return None

# --- Part 4: Main Orchestration ---
def main():
    """Main function to run the crawler and LLM processor."""
    parser = argparse.ArgumentParser(
        description="Crawl specific URLs from a sitemap, process with an LLM, and save the output."
    )
    parser.add_argument("sitemap_url", help="The full URL of the sitemap.xml file.")
    parser.add_argument("url_prefix", help="The URL prefix to filter which pages to crawl.")
    parser.add_argument(
        "--service", 
        choices=['openai', 'openrouter'], 
        default='openai',
        help="The LLM service to use (default: openai)."
    )
    parser.add_argument(
        "-o", "--output", 
        default="knowledge_base.md", 
        help="The name of the output file (default: knowledge_base.md)."
    )
    
    args = parser.parse_args()

    # Step 1: Get all URLs from the sitemap matching the prefix
    urls_to_crawl = get_urls_from_sitemap(args.sitemap_url, args.url_prefix)

    if not urls_to_crawl:
        print("[!] No URLs found matching the prefix. Exiting.")
        return

    # Step 2: Loop through URLs, crawl, process, and save
    print(f"[*] Starting to process {len(urls_to_crawl)} URLs. Output will be saved to '{args.output}'.")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, url in enumerate(urls_to_crawl, 1):
            print(f"\n--- Processing URL {i}/{len(urls_to_crawl)} ---")
            
            # Step 2a: Crawl the page
            knowledge_item = crawl_page_content(url)
            
            if knowledge_item:
                # Step 2b: Send to LLM for structuring
                structured_content = get_structured_content_from_llm(knowledge_item, args.service)
                
                if structured_content:
                    # Step 2c: Write to file
                    f.write(structured_content)
                    f.write("\n\n---\n\n") # Add a separator for readability
                    print(f"    [+] Successfully processed and saved content from {url}.")
    
    print(f"\n[***] All done! The structured knowledge base is saved in '{args.output}'. [***]")


if __name__ == "__main__":
    main()