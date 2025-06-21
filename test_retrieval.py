# test_retrieval.py

import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
import voyageai
import argparse
from typing import Optional

# --- 1. Configuration and Client Setup ---
load_dotenv()

# Voyage AI client for embedding user queries
VOYAGE_MODEL = "voyage-3-large"
try:
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    vo = voyageai.Client(api_key=voyage_api_key)
except Exception as e:
    raise ValueError(f"Voyage AI client initialization failed: {e}")

# Supabase client for database retrieval
try:
    url = os.getenv("SUPABASE_URL")
    # Use the SERVICE_ROLE_KEY as it has read access and bypasses RLS if needed for a script
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    supabase: Client = create_client(url, key)
except Exception as e:
    raise ValueError(f"Supabase client initialization failed: {e}")

# --- 2. The Retrieval Function (The Tool We Are Testing) ---
def retrieve_info(query: str, page_title_filter: Optional[str] = None) -> str:
    """
    Performs a hybrid search on the Supabase vector database.
    This is the core function we are evaluating.
    """
    print(f"\n> EXECUTING SEARCH for query='{query}'...")
    try:
        # Embed the query
        query_embedding = vo.embed([query], model=VOYAGE_MODEL, input_type="query").embeddings[0]
        
        # Define parameters for the RPC call to our SQL function
        rpc_params = {
            'query_embedding': query_embedding,
            'match_count': 3,                 # Get the top 3 results
            'similarity_threshold': 0.5,      # Filter out results with low semantic similarity
            'keyword': query                  # Use the query for keyword search as well
        }
        
        # Call the Supabase function
        response = supabase.rpc('match_dementia_chunks', rpc_params).execute()
        
        results = response.data
        if not results:
            return json.dumps({"status": "not_found", "message": "No information found."}, indent=2)
            
        return json.dumps(results, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"An error occurred during retrieval: {e}"}, indent=2)

# --- 3. Main Execution Block for Testing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the retrieval function for the RAG system.")
    parser.add_argument(
        "query", 
        help="The test question to send to the retrieval function, enclosed in quotes."
    )
    args = parser.parse_args()

    print("="*50)
    print("      PHASE 1: RETRIEVAL SYSTEM TEST      ")
    print("="*50)
    
    # Call the function with the provided query
    retrieved_data_json = retrieve_info(query=args.query)
    
    print("\n--- RETRIEVED CHUNKS (RAW JSON) ---")
    print(retrieved_data_json)
    print("="*50)