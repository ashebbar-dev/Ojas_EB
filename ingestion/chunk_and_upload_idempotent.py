# embed_and_upload_final.py

import json
import os
import voyageai
from supabase import create_client, Client
from dotenv import load_dotenv
import argparse

# --- Configuration ---
load_dotenv()
# Make sure to use the model that matches your SQL dimension setting
VOYAGE_MODEL = "voyage-3-large" # Or voyage-large-3-instruct etc.
VOYAGE_DIMENSIONS = 1024 # Must match the vector(1024) in your SQL

# --- Client Setup ---
# Voyage AI Client
voyage_api_key = os.getenv("VOYAGE_API_KEY")
if not voyage_api_key:
    raise ValueError("VOYAGE_API_KEY not found in .env file.")
vo = voyageai.Client(api_key=voyage_api_key)

# Supabase Client
url = os.getenv("SUPABASE_URL")
# IMPORTANT: Use the SERVICE_ROLE_KEY to bypass RLS for server-side operations
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") 
if not url or not key:
    raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in your .env file.")
supabase: Client = create_client(url, key)

def create_vector_database(json_file_path: str):
    """
    Reads the final JSON file, generates embeddings with Voyage, and populates
    the final, structured 'dementia_chunks' Supabase table.
    """
    print(f"[*] Loading refined chunks from '{json_file_path}'...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"[*] Preparing to embed and upload {len(chunks)} chunks to Supabase table 'dementia_chunks'...")

    batch_size = 50 # A good batch size for Voyage API
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        # We now call this 'batch_content' to match the column name 'content'
        batch_content = [chunk['text'] for chunk in batch_chunks] 
        
        print(f"    -> Processing batch {i//batch_size + 1} of {len(chunks)//batch_size + 1}...")

        # Get embeddings from Voyage AI
        result = vo.embed(batch_content, model=VOYAGE_MODEL, input_type="document")
        embeddings = result.embeddings

        # --- THIS IS THE UPDATED MAPPING LOGIC ---
        # We now map the data from the JSON file to the specific columns
        # in our new, improved database table.
        records_to_insert = []
        for j, chunk in enumerate(batch_chunks):
            metadata = chunk['metadata']
            record = {
                'content': chunk['text'],
                'source_url': metadata.get('source_url'),
                'page_title': metadata.get('page_title'),
                'topic_heading': metadata.get('topic_heading'),
                # We map the 'contents' key from our JSON to the 'processing_info' column
                'processing_info': metadata.get('contents'), 
                'embedding': embeddings[j]
            }
            records_to_insert.append(record)
        # --- END OF MAPPING LOGIC ---

        # Insert the formatted batch into the 'dementia_chunks' table in Supabase
        try:
            supabase.table('dementia_chunks').insert(records_to_insert).execute()
        except Exception as e:
            print(f"    [!] Error inserting batch into Supabase: {e}")
            # You might want more robust error handling here, but for now we stop
            return

    print("\n[***] Success! Your Supabase table 'dementia_chunks' is populated. Your data pipeline is complete. [***]")


def main():
    parser = argparse.ArgumentParser(description="Embed and upload a knowledge base to a production Supabase schema.")
    parser.add_argument("input_file", help="The final JSON file from refine_chunks.py (e.g., final_knowledge_base.json).")
    args = parser.parse_args()
    create_vector_database(args.input_file)

if __name__ == "__main__":
    main()