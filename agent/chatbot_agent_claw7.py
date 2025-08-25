import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from supabase import create_client, Client
import voyageai
import cohere
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import argparse
from typing import Optional, List, Dict, Any
import time
from collections import defaultdict

# --- LangChain Agent Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# ── NEW: callback that puts every token into a Queue ──────────────
# ── put this with the other imports (queue already imported earlier) ──
from queue import Queue
from langchain.callbacks.base import BaseCallbackHandler


# ------------------------------------------------------------------
#  Replace your previous SSECallbackHandler with THIS version
# ------------------------------------------------------------------
from queue import Queue
from langchain.callbacks.base import BaseCallbackHandler


class SSECallbackHandler(BaseCallbackHandler):
    """
    Put every new token into a Queue *after* the model has produced
    the marker text (default: “Final Answer:”).
    The marker itself is NOT forwarded.
    """
    def __init__(self, token_queue: Queue, start_marker: str = "Final Answer:"):
        self.q            = token_queue
        self.marker       = start_marker
        self.buffer       = ""      # accumulates tokens until marker shows up
        self.started      = False   # becomes True once we're past the marker
        self.marker_len   = len(start_marker)

    def on_llm_new_token(self, token: str, **kwargs):
        if self.started:
            # Already past marker → stream token immediately
            self.q.put(token)
            return

        # Still waiting: build up the buffer
        self.buffer += token

        if self.marker in self.buffer:
            # Split at first occurrence; keep ONLY the part *after* the marker
            after_marker = self.buffer.split(self.marker, 1)[1]
            if after_marker:
                self.q.put(after_marker)
            self.started = True
            # From now on subsequent tokens are streamed directly

# --- Configuration and Client Setup ---
load_dotenv()
VOYAGE_MODEL = "voyage-3-large"
COHERE_RERANK_MODEL = "rerank-english-v3.0"

try:
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    vo = voyageai.Client(api_key=voyage_api_key)
    
    cohere_api_key = os.getenv("COHERE_API_KEY")
    co = cohere.Client(api_key=cohere_api_key)
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    supabase: Client = create_client(url, key)
except Exception as e:
    raise ValueError(f"Initialization failed: {e}")

# --- Individual Search Function ---
def execute_dual_track_search(query: str, search_id: str = None) -> Dict[str, Any]:
    """Execute dual-track search for a single query"""
    search_start_time = time.time()
    print(f"🔍 [{search_id or 'SEARCH'}] Starting: '{query}'")
    
    try:
        # Generate query embedding
        query_embedding = vo.embed([query], model=VOYAGE_MODEL, input_type="query").embeddings[0]
        
        # Prepare search parameters
        search_params = {
            'query_embedding': query_embedding,
            'keyword': query,
            'match_count': 30,  # Get more for better reranking
            'similarity_threshold': 0.40  # Lower threshold for more recall
        }
        
        # Execute both searches in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Track 1: Simple hybrid search
            future_simple = executor.submit(
                lambda: supabase.rpc('simple_hybrid_search', search_params).execute()
            )
            
            # Track 2: Title-filtered search
            title_params = {**search_params, 'title_match_count': 10}
            future_title = executor.submit(
                lambda: supabase.rpc('title_filtered_search', title_params).execute()
            )
            
            # Get results
            simple_results = future_simple.result().data or []
            title_results = future_title.result().data or []
        
        # Combine and deduplicate results
        combined_results = []
        seen_ids = set()
        
        # Add results from both tracks
        for result in simple_results + title_results:
            if result['id'] not in seen_ids:
                combined_results.append(result)
                seen_ids.add(result['id'])
        
        # Rerank using Cohere if we have results
        if combined_results:
            print(f"🔄 [{search_id or 'SEARCH'}] Reranking {len(combined_results)} chunks with Cohere...")
            
            # Prepare documents for reranking
            documents = [result['content'] for result in combined_results]
            
            try:
                # Rerank with Cohere
                rerank_response = co.rerank(
                    model=COHERE_RERANK_MODEL,
                    query=query,
                    documents=documents,
                    top_n=min(10, len(documents)),  # Top 8 after reranking
                    return_documents=True
                )
                
                # Map reranked results back to original data
                reranked_results = []
                for idx, rerank_result in enumerate(rerank_response.results):
                    original_result = combined_results[rerank_result.index]
                    original_result['cohere_score'] = rerank_result.relevance_score
                    original_result['rerank_position'] = idx + 1
                    reranked_results.append(original_result)
                
                final_results = reranked_results
                
            except Exception as cohere_error:
                print(f"⚠️  Cohere reranking failed: {cohere_error}")
                print("🔄 Falling back to similarity-based ranking...")
                # Fallback: sort by similarity score
                final_results = sorted(combined_results, 
                                     key=lambda x: x.get('similarity', 0), 
                                     reverse=True)[:8]
        else:
            final_results = []
        
        search_time = time.time() - search_start_time
        print(f"✅ [{search_id or 'SEARCH'}] Completed in {search_time:.2f}s - {len(final_results)} results")
        
        return {
            "query": query,
            "search_id": search_id,
            "results": final_results,
            "search_time": search_time,
            "total_before_rerank": len(combined_results),
            "total_after_rerank": len(final_results)
        }
        
    except Exception as e:
        error_msg = f"Search failed for '{query}': {str(e)}"
        print(f"❌ [{search_id or 'SEARCH'}] {error_msg}")
        return {
            "query": query,
            "search_id": search_id,
            "error": error_msg,
            "results": [],
            "search_time": time.time() - search_start_time
        }

# --- ENHANCED Parallel Multi-Query Search Tool with Advanced Deduplication ---
@tool
def parallel_comprehensive_search(query_input: str) -> str:
    """
    Performs parallel comprehensive searches for multiple sub-queries.
    Enhanced with advanced deduplication that tracks which queries retrieved each chunk.
    Input can be either:
    1. A JSON string with a list: '["query1", "query2"]'
    2. A single query string: 'single query'
    3. A JSON object: '{"queries_json": "[\"query1\", \"query2\"]"}'
    
    Returns comprehensive results with deduplication tracking.
    """
    print(f"\n🔍 RAW INPUT: {query_input}")
    
    # IMPROVED: Handle multiple input formats
    queries = []
    
    try:
        # First, try to parse as JSON
        parsed = json.loads(query_input)
        
        if isinstance(parsed, list):
            # Format 1: Direct list ["query1", "query2"]
            queries = parsed
        elif isinstance(parsed, dict) and 'queries_json' in parsed:
            # Format 3: Nested object {"queries_json": "[\"query1\", \"query2\"]"}
            inner_json = parsed['queries_json']
            queries = json.loads(inner_json) if isinstance(inner_json, str) else inner_json
        elif isinstance(parsed, str):
            # Format 2: Single string or double-encoded JSON
            try:
                queries = json.loads(parsed)
            except:
                queries = [parsed]
        else:
            queries = [str(parsed)]
            
    except (json.JSONDecodeError, TypeError, KeyError):
        # If all JSON parsing fails, treat as single query
        print("⚠️  JSON parsing failed, treating as single query")
        queries = [query_input.strip()]
    
    # Validation and cleanup
    queries = [q.strip() for q in queries if q.strip()]
    if not queries:
        queries = ["general dementia care information"]
    
    print(f"\n🚀 PARALLEL SEARCH: {len(queries)} sub-queries")
    for i, query in enumerate(queries, 1):
        print(f"   {i}. {query}")
    
    overall_start_time = time.time()
    
    # Execute all searches in parallel
    with ThreadPoolExecutor(max_workers=min(len(queries), 4)) as executor:
        # Submit all search tasks
        future_to_query = {
            executor.submit(execute_dual_track_search, query, f"Q{i+1}"): query 
            for i, query in enumerate(queries)
        }
        
        # Collect results as they complete
        all_search_results = []
        for future in as_completed(future_to_query):
            result = future.result()
            all_search_results.append(result)
    
    # Sort results by original query order
    search_id_to_order = {f"Q{i+1}": i for i in range(len(queries))}
    all_search_results.sort(key=lambda x: search_id_to_order.get(x.get('search_id', 'Q999'), 999))
    
    # ENHANCED DEDUPLICATION: Group chunks by ID and track all source queries
    chunk_groups = defaultdict(lambda: {
        'chunk_data': None,
        'source_queries': [],
        'source_query_ids': [],
        'best_scores': {'similarity': 0, 'cohere_score': 0}
    })
    
    print(f"\n🔄 ENHANCED DEDUPLICATION: Processing chunks...")
    
    for search_result in all_search_results:
        query_info = {
            "query": search_result["query"],
            "search_id": search_result.get("search_id"),
            "results_count": len(search_result["results"]),
            "search_time": search_result.get("search_time", 0),
            "error": search_result.get("error")
        }
        
        # Process each chunk from this query
        for result in search_result["results"]:
            chunk_id = result['id']
            
            # If this is the first time we see this chunk, store it
            if chunk_groups[chunk_id]['chunk_data'] is None:
                chunk_groups[chunk_id]['chunk_data'] = result.copy()
            
            # Add this query to the sources for this chunk
            chunk_groups[chunk_id]['source_queries'].append(search_result["query"])
            chunk_groups[chunk_id]['source_query_ids'].append(search_result.get("search_id"))
            
            # Track best scores across all queries for this chunk
            similarity = result.get('similarity', 0)
            cohere_score = result.get('cohere_score', 0)
            
            if similarity > chunk_groups[chunk_id]['best_scores']['similarity']:
                chunk_groups[chunk_id]['best_scores']['similarity'] = similarity
            if cohere_score > chunk_groups[chunk_id]['best_scores']['cohere_score']:
                chunk_groups[chunk_id]['best_scores']['cohere_score'] = cohere_score
    
    # Convert grouped chunks back to list format
    deduplicated_chunks = []
    for chunk_id, group_data in chunk_groups.items():
        chunk = group_data['chunk_data']
        if chunk:
            # Add deduplication metadata
            chunk['retrieved_by_queries'] = list(set(group_data['source_queries']))  # Remove duplicate queries
            chunk['retrieved_by_query_ids'] = list(set(group_data['source_query_ids']))
            chunk['retrieval_count'] = len(group_data['source_queries'])  # Total retrievals (with duplicates)
            chunk['unique_query_count'] = len(set(group_data['source_queries']))  # Unique queries that found it
            
            # Use best scores across all retrievals
            chunk['best_similarity'] = group_data['best_scores']['similarity']
            chunk['best_cohere_score'] = group_data['best_scores']['cohere_score']
            
            deduplicated_chunks.append(chunk)
    
    print(f"📊 DEDUPLICATION RESULTS:")
    print(f"   • Total chunks before deduplication: {sum(len(sr['results']) for sr in all_search_results)}")
    print(f"   • Unique chunks after deduplication: {len(deduplicated_chunks)}")
    
    # Final reranking of deduplicated results if we have multiple queries
    if len(queries) > 1 and deduplicated_chunks:
        print(f"🔄 FINAL RERANK: {len(deduplicated_chunks)} deduplicated chunks")
        
        # Use the combined queries for final reranking
        main_query = " ".join(queries)
        documents = [chunk['content'] for chunk in deduplicated_chunks]
        
        try:
            final_rerank = co.rerank(
                model=COHERE_RERANK_MODEL,
                query=main_query,
                documents=documents,
                top_n=min(10, len(documents)),  # Top 12 overall
                return_documents=True
            )
            
            # Apply final ranking
            final_chunks = []
            for idx, rerank_result in enumerate(final_rerank.results):
                chunk = deduplicated_chunks[rerank_result.index]
                chunk['final_cohere_score'] = rerank_result.relevance_score
                chunk['final_rank'] = idx + 1
                final_chunks.append(chunk)
        
        except Exception as e:
            print(f"⚠️  Final reranking failed: {e}")
            final_chunks = deduplicated_chunks[:12]  # Fallback to top 12
    else:
        final_chunks = deduplicated_chunks[:8]  # Single query case
    
    # ENHANCED: Format results with deduplication information
    formatted_results = []
    for chunk in final_chunks:
        formatted_chunk = {
            "chunk_id": chunk.get("id"),  # Include chunk ID for debugging
            "source_url": chunk.get("source_url"),
            "page_title": chunk.get("page_title"), 
            "topic_heading": chunk.get("topic_heading"),
            "content": chunk.get("content")[:800] + "..." if len(chunk.get("content", "")) > 800 else chunk.get("content", ""),
            
            # Scoring information
            "similarity_score": round(chunk.get("similarity", 0), 3),
            "cohere_score": round(chunk.get("cohere_score", 0), 3),
            "best_similarity": round(chunk.get("best_similarity", 0), 3),
            "best_cohere_score": round(chunk.get("best_cohere_score", 0), 3),
            "final_cohere_score": round(chunk.get("final_cohere_score", 0), 3),
            "final_rank": chunk.get("final_rank"),
            
            # ENHANCED: Deduplication tracking
            "retrieved_by_queries": chunk.get("retrieved_by_queries", []),
            "retrieved_by_query_ids": chunk.get("retrieved_by_query_ids", []),
            "retrieval_count": chunk.get("retrieval_count", 1),
            "unique_query_count": chunk.get("unique_query_count", 1),
            "is_duplicate_found": chunk.get("retrieval_count", 1) > 1
        }
        formatted_results.append(formatted_chunk)
    
    # Build comprehensive summary
    search_summary = {
        "total_queries": len(queries),
        "parsed_queries": queries,
        "deduplication_stats": {
            "total_chunks_before": sum(len(sr['results']) for sr in all_search_results),
            "unique_chunks_after": len(deduplicated_chunks),
            "duplicates_removed": sum(len(sr['results']) for sr in all_search_results) - len(deduplicated_chunks),
            "chunks_found_multiple_times": sum(1 for chunk in final_chunks if chunk.get("retrieval_count", 1) > 1)
        },
        "query_details": [
            {
                "query": sr["query"],
                "search_id": sr.get("search_id"),
                "results_count": len(sr["results"]),
                "search_time": sr.get("search_time", 0),
                "error": sr.get("error")
            } for sr in all_search_results
        ],
        "results": formatted_results,
        "total_results": len(formatted_results),
        "overall_time": round(time.time() - overall_start_time, 2)
    }
    
    overall_time = time.time() - overall_start_time
    print(f"✅ PARALLEL SEARCH COMPLETED: {overall_time:.2f}s total")
    print(f"📊 FINAL STATS: {len(formatted_results)} unique chunks, {search_summary['deduplication_stats']['duplicates_removed']} duplicates removed")
    
    if not formatted_results:
        return json.dumps({
            "status": "not_found",
            "message": "No relevant information found for any of the queries.",
            "queries": queries,
            "search_summary": search_summary
        })
    
    return json.dumps(search_summary, indent=2)

# --- SIMPLIFIED Agent Prompt ---
simplified_react_prompt_template = """
You are OJAS, a 'Caregiver Companion,' an expert AI assistant. Your purpose is to provide clear, empathetic, and actionable answers to caregivers of people with dementia. You do this by searching for relevant information in the database.
Only answer questions related to dementia care or politely decline.
Questions related to dementia should only be answered after searching the database.
Provide an in-depth answer; if the question is simple, try to cover related topics.

**YOUR APPROACH:**
- Think step by step.
1. **ANALYZE** the user's question deeply: Identify the main topic, any sub-aspects (e.g., symptoms, management, prevention), user context (e.g., relationship like 'mom', specific dementia type), and potential related topics (e.g., safety tips, emotional support).
2. **GENERATE SUB-QUERIES**: Based on the analysis, create aligned sub-queries that complement each other for synergy (e.g., one for causes, one for management). Ensure they build on the core theme and synergize to provide holistic coverage without redundancy.Your knowledge base is exclusively from the Alzheimer's Society. Create 2-5 specific, targeted sub-queries that cover all aspects of the user's question. Frame these queries as if you were searching for articles or help pages on a professional dementia support website like the Alzheimer's Society. Make them precise and dementia-focused (e.g., include 'in dementia' or 'for caregivers'). Avoid vagueness or repetition. Format as a simple JSON array.
3. **SEARCH** using parallel_comprehensive_search with ALL sub-queries in one JSON array.
4. **SYNTHESIZE** the results into a comprehensive, helpful answer.

**SUB-QUERY GUIDELINES**:
- Be specific: Instead of 'wandering', use 'managing nighttime wandering in dementia patients'.
- Cover breadth: Include queries for definitions, causes, strategies, and related risks.
- Limit to 3-5: Prioritize the most relevant.
- Adapt to user: If they mention 'Alzheimer's', include it in sub-queries.

**SEARCH FORMAT:**
Use this exact format for the tool:
["specific query 1", "specific query 2", "specific query 3"]

**CRITICAL RULES:**
- Answer only dementia care questions.
- Put ALL sub-queries in one JSON array.
- Always interpret any dementia-related statement as an implicit request for information or support, and generate sub-queries based on possible concerns.
- Use the tool ONCE with all queries.
- Your FINAL response must start with "Final Answer:" and use markdown headings (e.g., ### Definition of Dementia).
- Stop generating immediately after "Action Input:" - do NOT write "Observation:" or "Final Answer:" after tool use.
- Do not write "**" for Thought, Action or Action Input 
- Think step by step.

For greetings like "hi", answer directly with "Final Answer:".

**TOOLS:**
{tools}

<!-- Examples below are ONLY for your reference, not part of the live chat -->
**GOOD EXAMPLE:**

Question: How can I help my mom with eating problems and wandering at night?

Thought: Analyze: Main aspects are eating problems (e.g., swallowing, appetite) and nighttime wandering (e.g., safety, prevention). User context: 'mom' implies elderly female. Related topics: behavioral strategies, caregiver tips. Generate 4 specific sub-queries covering definitions, management, and prevention.
Action: {tool_names}
Action Input: ["managing eating and swallowing difficulties in dementia", "preventing nighttime wandering in elderly dementia patients", "caregiver strategies for dementia behavioral issues", "related safety tips for dementia at home"]

**BAD EXAMPLE (TO AVOID):**
Thought: Too vague sub-queries.
Action Input: ["eating problems", "wandering", "dementia"]

Begin!

Question: {input}
{agent_scratchpad}
"""

simplified_react_prompt = ChatPromptTemplate.from_template(simplified_react_prompt_template)

# --- Enhanced Chatbot Function ---
def ask_parallel_enhanced_chatbot(
        user_query: str = None,
        messages: list  | None = None,    # ← added
        service: str = "openrouter",
        model_name: str = "qwen/qwen-2.5-72b-instruct",
        stream_answer: bool = False,
        token_queue: Queue | None = None,
):
    """
    Runs the enhanced chatbot with parallel sub-query execution, Cohere reranking, and advanced deduplication.
    """
    if messages is None and user_query is None:
        raise ValueError("Provide either 'user_query' or 'messages'.")
    
    print("=" * 70)
    print(f"🚀 PARALLEL ENHANCED CAREGIVER COMPANION")
    print(f"Query: {user_query}")
    print(f"Service: {service} | Model: {model_name}")
    print(f"Features: Dual-Track Search + Cohere Rerank + Parallel Exec + "f"Adv. Deduplication{' + Streaming' if stream_answer else ''}")
    print("=" * 70)


    # Configure LLM with stop sequences
    stop_sequence = ["\nObservation:", "Observation:"]

    if stream_answer:
        callbacks = ([SSECallbackHandler(token_queue)]
                    if token_queue is not None
                    else [StreamingStdOutCallbackHandler()])
    else:
        callbacks = None

    if service == 'openrouter':
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables.")
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.20,
            stop=stop_sequence,
            streaming=stream_answer,      # <── HERE
            callbacks=callbacks
        )
    elif service == 'groq':
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        llm = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=0.2,
            stop=stop_sequence,
            streaming=stream_answer,      # <── HERE
            callbacks=callbacks 
        )
    else:
        raise ValueError(f"Unsupported service: {service}")

    # Create and run the enhanced agent
    tools = [parallel_comprehensive_search]
    agent = create_react_agent(llm, tools, simplified_react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5  # Keep it simple
    )

    # ------------------------------------------------------------------
# Invoke with messages or with plain 'input'
# ------------------------------------------------------------------
    if messages is not None:
        response = agent_executor.invoke({"messages": messages})
    else:
        response = agent_executor.invoke({"input": user_query})
    final_answer = response.get("output", "I apologize, but I couldn't generate a complete response.")
    

    # Display final result
    if stream_answer:
        print("\n" + "-"*60)
    print("\n" + "=" * 30 + " FINAL RESPONSE " + "=" * 30)
    print("⚠️  Disclaimer: I am an AI assistant, not a medical professional. Please consult healthcare providers for medical concerns.\n")
    print(final_answer)
    print("=" * 75)
    return final_answer           #  ← add this

# --- Command Line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel enhanced RAG agent with streaming option"
    )
    parser.add_argument("--service", choices=['openrouter', 'groq'], required=True,
                        help="LLM service to use")
    parser.add_argument("--query", required=True, help="User question in quotes")
    parser.add_argument("--stream", action="store_true",
                        help="Stream the assistant's final answer token-by-token")
    args = parser.parse_args()

    model_to_use = "mistralai/mistral-small-3.2-24b-instruct"
    if args.service == 'groq':
        model_to_use = "llama3-70b-8192"

    ask_parallel_enhanced_chatbot(
        user_query=args.query,
        service=args.service,
        model_name=model_to_use,
        stream_answer=args.stream        # <── PASS FLAG
    )