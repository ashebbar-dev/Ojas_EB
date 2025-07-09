# OJAS -  Advanced RAG Agent for Dementia Caregivers

This repository contains the complete source code and documentation for a state-of-the-art, multi-path Retrieval-Augmented Generation (RAG) agent. It is designed to function as an empathetic, accurate, and verifiable AI assistant for caregivers of people with dementia, using `alzheimers.org.uk` as its exclusive knowledge source.

This is not a simple RAG implementation. It is an advanced, multi-stage system designed to overcome common RAG pitfalls by employing query deconstruction, parallel-path retrieval, state-of-the-art reranking, and a robust, streaming-first web interface.

---

## 🌟 Core Features & Technical Highlights

- **Sophisticated Agentic Workflow:** Implements a "Decompose -> Retrieve -> Synthesize" pipeline, allowing the agent to break down complex, multi-intent user queries into focused sub-queries before retrieving information.
- **Novel Parallel-Path Retrieval:** The core of the system is a custom retrieval tool that executes two search strategies in parallel for each sub-query:
    1.  **Broad Hybrid Search:** A "wide net" combining semantic (vector) and lexical (keyword) search to maximize recall across the entire knowledge base.
    2.  **Title-First Entity Search:** A "sniper rifle" that first identifies the most relevant articles by title and then retrieves their full context, maximizing precision.
- **State-of-the-Art Reranking:** Utilizes Cohere's `rerank-english-v3.0` model as a final, crucial quality gate to re-order candidate chunks based on true contextual relevance, significantly improving the signal-to-noise ratio of the retrieved context.
- **Specialized & Robust Data Ingestion:** The data pipeline is not a one-size-fits-all solution. It uses multiple, specialized scripts tailored to handle the inconsistent HTML structures found across different sections of the source website (e.g., informational pages vs. blog posts).
- **Automated Content Curation & QC:** The pipeline automatically filters content by category (e.g., only "Advice," "Research") and flags pages with abnormal structures for manual review, ensuring a high-quality, relevant knowledge base.
- **Production-Grade Backend & UI:** A non-blocking Flask backend serves a real-time streaming API using Server-Sent Events (SSE). The vanilla JS/Tailwind CSS frontend consumes this stream, providing an immediate, token-by-token response for an excellent user experience.
- **Secure & Performant Database:** Built on Supabase (PostgreSQL with `pgvector`), the database schema is fully indexed for vector search (`ivfflat`), full-text search (`gin`), and metadata filtering (`btree`), ensuring high performance at scale.

---

## 🏗️ System Architecture Deep Dive

The agent operates on a sophisticated pipeline designed for maximum reliability and answer quality.

1.  **User Interface (`ui/index.html`):** A user submits a query through the web interface.
2.  **API Server (`agent/server.py`):** The Flask server receives the query at the `/ask_stream` endpoint. It dispatches the main agent task to a background thread to keep the server responsive.
3.  **Decomposition (`agent/chatbot_agent_claw4.py`):** The agent's first action is a dedicated LLM call. It uses a "Strategist" prompt to analyze the user's query and break it down into a list of focused, self-contained sub-queries.
    - *Example:* "How do I handle my dad's wandering and eating problems?" -> `["managing wandering behavior in dementia", "addressing eating problems in dementia"]`
4.  **Parallel Retrieval (`agent/chatbot_agent_claw4.py`):** The agent makes a single call to the `parallel_comprehensive_search` tool, passing the entire list of sub-queries.
    - This tool uses a `ThreadPoolExecutor` to run a `execute_dual_track_search` worker for each sub-query simultaneously.
    - Each worker executes two database searches in parallel: `simple_hybrid_search` and `title_filtered_search`.
    - The results from both paths are combined, de-duplicated, and put through Cohere Reranking.
5.  **Meta-Reranking & Context Assembly:** The results from all parallel workers are collected. A final reranking pass is performed on this entire pool of context to find the most relevant chunks related to the user's *original* query.
6.  **Synthesis & Streaming:** The final, high-quality context is fed into a dedicated "Synthesizer" LLM call. This LLM's only job is to write a comprehensive, empathetic answer. As it generates tokens, a custom `SSECallbackHandler` puts them into a queue, which the Flask server streams directly to the user's browser.

---

## 🛠️ Setup and Installation Guide

Follow these steps meticulously to set up the project environment and run the application.

### 1. Prerequisites

- Python 3.10 or newer.
- An account on [Supabase](https://supabase.com) to create a new PostgreSQL project.
- API keys from the following services:
    - [Voyage AI](https://www.voyageai.com/) (for embeddings)
    - [Cohere](https://cohere.com/) (for reranking)
    - [OpenRouter](https://openrouter.ai/) or [Groq](https://groq.com/) (for LLM access)

### 2. Environment Setup

**Clone the Repository:**
```bash
git clone https://github.com/enigmatulipgarde00n/Ojas_EB.git
cd Ojas_EB
```

**Create and Activate a Virtual Environment (Highly Recommended):**
A virtual environment isolates your project's dependencies from your system's global Python installation.

```bash
# Create the virtual environment (named 'venv')
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
You will see `(venv)` at the beginning of your terminal prompt, indicating it's active.

**Install Python Dependencies:**
Install all required libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**Set Up Environment Variables:**
This project uses a `.env` file to manage secret API keys.

1.  Copy the example file:
    ```bash
    cp .env.example .env
    ```
2.  Open the newly created `.env` file in a text editor.
3.  Fill in the placeholder values with your actual keys from Supabase, Voyage, Cohere, and your chosen LLM provider.

### 3. Database & Data Ingestion Pipeline

This multi-step process populates your Supabase database with the knowledge base. Run these steps in order.

**Step 3.1: Initialize the Database Schema**
- **Action:** Go to your Supabase project dashboard.
- **Navigate:** Find the "SQL Editor" in the left-hand menu.
- **Execute:** Open the `database/schema.sql` file from this repository, copy its entire content, paste it into the SQL Editor, and click "RUN".
- **Purpose:** This one-time setup creates the `dementia_chunks` table, all necessary performance indexes (`ivfflat`, `gin`), and the three custom SQL functions (`simple_hybrid_search`, `title_filtered_search`, `find_relevant_pages`) required for advanced retrieval.

**Step 3.2: Crawl and Chunk the Content**
The data is gathered using two specialized scripts to handle the website's different layouts.

- **Action 1: Crawl Informational Pages**
    ```bash
    python ingestion/trial_nollm_crawl1.py "https://www.alzheimers.org.uk/sitemap.xml" "https://www.alzheimers.org.uk/about-dementia" -o about_dementia.json
    ```
    - **Purpose:** This script specifically targets the main informational sections. It uses a simpler chunking strategy based on `<h2>` tags, which is optimal for these pages. It outputs `about_dementia.json`.

- **Action 2: Crawl and Curate Blog Pages**
    ```bash
    python ingestion/curated_chunker_with_log1.py "https://www.alzheimers.org.uk/sitemap.xml" "https://www.alzheimers.org.uk/blog" -o blog_posts.json
    ```
    - **Purpose:** This script targets the blog. It first verifies each article belongs to an approved category (e.g., "Advice") and then uses a different chunking strategy optimized for the blog's HTML structure. It also logs any pages with abnormal chunk counts to `outliers_for_review.csv` for quality control. It outputs `blog_posts.json`.

**Step 3.3: Refine and Prepare the Final Dataset**
- **Action 1: Combine Files:** Manually or programmatically merge the contents of `about_dementia.json` and `blog_posts.json` into a single file named `knowledge_base.json`.
- **Action 2: Refine Chunks**
    ```bash
    python ingestion/refine_chunks.py knowledge_base.json -o final_knowledge_base.json
    ```
    - **Purpose:** This critical script takes the structurally-chunked data and processes it for optimal LLM performance. It uses `tiktoken` to ensure every chunk is within a target token range (e.g., 300-600 tokens) by intelligently merging small chunks and splitting large ones.

**Step 3.4: Embed and Upload to Supabase**
- **Action:**
    ```bash
    python ingestion/embed_and_upload_idempotent.py final_knowledge_base.json
    ```
    - **Purpose:** This is the final ingestion step. The script is **idempotent**, meaning it first deletes all existing data from the `dementia_chunks` table to prevent duplicates. It then reads `final_knowledge_base.json`, generates embeddings for each chunk using the Voyage AI API, and uploads the content, metadata, and embeddings to your Supabase database in efficient batches.

Your knowledge base is now live and ready for querying.

### 4. Running the Chatbot Application

With the data pipeline complete, you can now start the application.

**Step 4.1: Start the Backend Server**
- **Action:**
    ```bash
    python agent/server.py
    ```
    - **Purpose:** This starts the Flask web server on `http://127.0.0.1:5000`. It listens for API requests from the frontend, manages chat sessions, and orchestrates the agent's work in background threads.

**Step 4.2: Open the User Interface**
- **Action:** Open the `ui/index.html` file in your web browser. You don't need to serve it; you can open it directly from your file system.
- **Interact:** The UI will connect to your local Flask server. You can now start a new chat and ask questions. Watch your terminal where the server is running to see the agent's detailed reasoning trace in real-time.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.



