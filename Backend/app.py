import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
import voyageai
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional

# --- LangChain Agent Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool

### --- CHANGE #1: Import Flask --- ###
from flask import Flask, request, jsonify
from flask_cors import CORS

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

# --- 2. Smart Tool Definition (Unchanged) ---
@tool
def retrieve_info(query: str, page_title_filter: Optional[str] = None) -> str:
    """
    Searches the dementia knowledge base for information. Use this to find answers to specific questions.
    You can optionally filter by 'page_title_filter' to narrow the search.
    """
    print(f"\n> TOOL CALL: Retrieving info for query='{query}' with filter='{page_title_filter}'")
    try:
        query_embedding = vo.embed([query], model=VOYAGE_MODEL, input_type="query").embeddings[0]
        rpc_params = {
            'query_embedding': query_embedding,
            'match_count': 5,
            'similarity_threshold': 0.5,
            'keyword': query
        }
        response = supabase.rpc('match_dementia_chunks', rpc_params).execute()
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

# --- 3. The "All-in-One" Agent's Brain (Unchanged) ---
react_prompt_template = """
1. You are the 'Caregiver Companion,' an expert AI assistant. Your purpose is to provide clear, empathetic, and actionable answers to caregivers of people with dementia.
2. make sure you have gathered information on all parts of user's query before answering

You will answer the user's question by following these steps:
1. You will have to gather all the relavent context to answer user's query.To gather context You can Use the `retrieve_info` tool to search your knowledge base for relevant information on questions you ask. Don't just ask what user asked directly to the database. reformulate user questions into high-quality search queries for a vector database. Gather context about 1 topic per turn. Analyze the information you've gathered from the tool calls. Do this multiple times to gather information about user's query and then when you have gathered all the information, provide the final answer.
2. make sure you have gathered information on all parts of user's query before answering

**CRITICAL RULES FOR RESPONDING:**
- You MUST follow a strict "Thought -> Action -> Action Input" cycle when using tools.
- **You MUST stop generating text immediately after you output the Action Input block. Do NOT generate the 'Observation:' part yourself. The system will provide it to you.**
- Your FINAL response to the user must be prefixed with the special string "Final Answer:".
- The final answer should be well-structured and supportive.

**TOOLS:**
------
You have access to the following tool:
{tools}

**HOW TO USE THE TOOL:**
------
Thought: Your reasoning for what information you need to find next.
Action: The name of the tool to use, which must be one of [{tool_names}].
Action Input: A JSON string with the tool's arguments.

---
**EXAMPLE OF A COMPLETE PROCESS:**

Question: My dad is having trouble with sleeping and gets restless. What can I do to help?

Thought: The user needs help with two issues: sleep problems and restlessness in their father with dementia. I will start by searching for information on sleep disturbances.
Action: retrieve_info
Action Input: {{
  "query": "strategies for managing sleep disturbances in dementia"
}}
Observation: [{{... "content": "Establish a regular bedtime... avoid caffeine... keep the bedroom quiet..." ...}}]

Thought: I have good information on sleep, but I need to specifically address the "restlessness" part. I will perform another search focused on that.
Action: retrieve_info
Action Input: {{
  "query": "managing restlessness and agitation in dementia"
}}
Observation: [{{... "content": "Restlessness can be caused by pain, boredom, or anxiety. Try calming activities, gentle exercise, and check for physical discomfort..." ...}}]

Thought: I have now gathered sufficient information on both sleep and restlessness. I can synthesize this into a complete, structured answer for the user. I will now provide the final answer.
Final Answer: I understand it can be challenging when a loved one is struggling with sleep and restlessness. Based on the information I found, here are several strategies that can help:

**1. Establish a Strong Sleep Routine:**
*   **Consistent Schedule:** Try to maintain a regular bedtime and wake-up time, even on weekends.
*   **Limit Naps:** If your dad naps during the day, keep them short and not too close to bedtime.
*   **Avoid Stimulants:** Reduce or eliminate caffeine and nicotine, especially in the evening.

**2. Create a Calm Environment:**
*   **Comfort is Key:** Ensure the bedroom is quiet, dark, and at a comfortable temperature.
*   **Safety First:** If he gets restless, make sure the area around the bed is safe by removing sharp objects or adding soft padding on the floor.

**3. Address Restlessness:**
*   **Encourage Daytime Activity:** Gentle physical activity, like a daily walk, can help reduce restlessness at night.
*   **Check for Discomfort:** Restlessness can sometimes be a sign of pain, hunger, or another physical need.
*   **Calming Activities:** Before bed, try calming activities like listening to soft music or reading aloud.

It's always a good idea to discuss these ongoing issues with his doctor to rule out any underlying medical causes. They can provide the most personalized advice.
---

Begin!

Question: {input}

{agent_scratchpad}
"""
react_prompt = ChatPromptTemplate.from_template(react_prompt_template)

### --- CHANGE #2: Initialize Flask App --- ###
app = Flask(__name__)
# CORS is crucial for allowing the browser to make requests to this server
CORS(app)


### --- CHANGE #3: Modify ask_chatbot and create the API endpoint --- ###
# This function is no longer called directly from the command line.
# It's now the core logic for our API endpoint.
def run_chatbot_logic(user_query: str):
    """
    This function now contains the core chatbot logic.
    It takes a query and RETURNS the final answer as a string.
    """
    # For the web UI, let's default to a reliable service and model.
    # You can make this configurable if you want.
    service = 'openrouter'
    model_name = 'qwen/qwen3-14b'

    print("="*50)
    print(f"User Query: {user_query}")
    print(f"Using Service: {service} with Model: {model_name}")
    print("="*50)

    stop_sequence = ["\nObservation:"]
    llm = None
    if service == 'openrouter':
        # ... (code for openrouter is fine, keeping for completeness)
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key: raise ValueError("OPENROUTER_API_KEY not found")
        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url="https://openrouter.ai/api/v1", temperature=0.0, stop=stop_sequence)
    elif service == 'groq':
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: raise ValueError("GROQ_API_KEY not found")
        llm = ChatGroq(model_name=model_name, api_key=api_key, temperature=0.0, stop=stop_sequence)
    else:
        raise ValueError(f"Unsupported service: {service}")

    tools = [retrieve_info]
    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=7)

    try:
        response = agent_executor.invoke({"input": user_query})
        agent_final_answer = response.get("output", "I'm sorry, I encountered an error and couldn't generate a response.")
    except Exception as e:
        print(f"Error during agent execution: {e}")
        agent_final_answer = "I'm having trouble connecting to my knowledge base right now. Please try again in a moment."

    # Instead of printing, we return the final answer
    return agent_final_answer

# This is our new API endpoint. The frontend will send requests here.
@app.route('/ask', methods=['POST'])
def ask():
    # Get the user's query from the JSON body of the request
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Run the chatbot logic
    final_answer = run_chatbot_logic(user_query)

    # Return the answer as JSON
    return jsonify({"answer": final_answer})


### --- CHANGE #4: Run the Flask server instead of the command-line interface --- ###
if __name__ == "__main__":
    # This starts the web server.
    # debug=True allows for auto-reloading when you save the file.
    app.run(host='0.0.0.0', port=5000, debug=True)