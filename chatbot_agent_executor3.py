# chatbot_agent_final_v7.py

import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
import voyageai
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import argparse
from typing import Optional, Union

# --- LangChain Agent Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool

# --- 1. Configuration and Client Setup (Unchanged) ---
load_dotenv()
VOYAGE_MODEL = "voyage-3-large"
try:
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    vo = voyageai.Client(api_key=voyage_api_key)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    supabase: Client = create_client(url, key)
except Exception as e:
    raise ValueError(f"Initialization failed: {e}")

# --- 2. Smart Tool Definitions (Unchanged) ---
@tool
def retrieve_info(query: str, page_title_filter: Optional[str] = None) -> str:
    """
    Searches the dementia knowledge base for information. Use this to find answers to specific questions.
    You can optionally filter by 'page_title_filter' to narrow the search.
    """
    print(f"\n> TOOL CALL: Retrieving info for query='{query}' with filter='{page_title_filter}'")
    try:
        query_embedding = vo.embed([query], model=VOYAGE_MODEL, input_type="query").embeddings[0]
        
        # --- THIS IS THE UPDATED RPC CALL ---
        rpc_params = {
            'query_embedding': query_embedding,
            'match_count': 3,
            'similarity_threshold': 0.5, # Set a reasonable threshold
            'keyword': query # Use the original query for keyword search
        }
        # The filter logic is no longer needed here as the hybrid search is more robust
        # but you could add it back if you wanted to filter the keyword search too
        
        response = supabase.rpc('match_dementia_chunks', rpc_params).execute()
        # --- END OF UPDATED RPC CALL ---
        
        results = response.data
        if not results:
            return json.dumps({"status": "not_found", "message": "No information found."})
            
        simplified_results = []
        for res in results:
            simplified_results.append({
                "source_url": res.get("source_url"),
                "page_title": res.get("page_title"),
                "topic_heading": res.get("topic_heading"),
                "content": res.get("content")
            })
        return json.dumps(simplified_results, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Retrieval failed: {e}"})

        
@tool
def finish(answer: str) -> str:
    """
    Call this tool when you have gathered all necessary information and are ready to provide the final answer.
    The 'answer' parameter should be your complete, synthesized response.
    """
    return answer

# --- 3. The Agent's "Brain": The FINAL, Corrected Prompt ---
# THIS IS THE KEY FIX. We are extremely explicit about the JSON format for Action Input.
react_prompt_template = """
You are the 'Caregiver Companion,' an expert AI assistant. Your SOLE PURPOSE is to answer the user's question by gathering information from a knowledge base using the provided tools.

**CRITICAL RULES:**
- You MUST follow a strict "Thought-Action-Observation" cycle.
- You MUST NOT answer questions from your own knowledge.
- Your final answer MUST be provided by using the 'finish' tool.

**TOOLS:**
------
You have access to the following tools:

{tools}

To use a tool, you MUST use the following format, with the Action Input being a valid JSON object:
Thought: Your reasoning and plan for the next step.
Action: The name of the tool to use, which must be one of [{tool_names}]
Action Input: A JSON object with the arguments for the tool. For example: {{"query": "what is dementia"}} or {{"answer": "This is the final answer."}}

When you have gathered enough information to answer the question, you MUST use the 'finish' tool.

**Begin your work now.**

Question: {input}

{agent_scratchpad}
"""
react_prompt = ChatPromptTemplate.from_template(react_prompt_template)

# --- 4. Main Chatbot Function (Unchanged) ---
def ask_chatbot(user_query: str, service: str, model_name: str):
    # ... (the rest of the script is identical) ...
    print("="*50)
    print(f"User Query: {user_query}")
    print(f"Using Service: {service} with Model: {model_name}")
    print("="*50)
    llm = None
    if service == 'openrouter':
        api_key = os.getenv("OPENROUTER_API_KEY")
        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url="https://openrouter.ai/api/v1", temperature=0.1)
    elif service == 'groq':
        api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(model_name=model_name, api_key=api_key, temperature=0.1)
    else:
        raise ValueError(f"Unsupported service: {service}")
    tools = [retrieve_info, finish]
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=7
    )
    response = agent_executor.invoke({"input": user_query})
    agent_final_answer = response.get("output", "The agent did not call the finish tool correctly.")
    print("\n\n" + "="*20 + " CAREGIVER COMPANION RESPONSE " + "="*20)
    print("Disclaimer: I am an AI assistant and not a medical professional. Please consult a doctor for health concerns.\n")
    print(agent_final_answer)
    print("="*64)

# --- 5. Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An advanced, transparent RAG agent for dementia caregivers.")
    parser.add_argument("--service", choices=['openrouter', 'groq'], required=True, help="The LLM service to use.")
    parser.add_argument("--query", required=True, help="The question to ask the chatbot, enclosed in quotes.")
    args = parser.parse_args()
    if args.service == 'openrouter':
        model_to_use = "qwen/qwen3-14b"
    elif args.service == 'groq':
        model_to_use = "llama3-70b-8192"
    ask_chatbot(user_query=args.query, service=args.service, model_name=model_to_use)