# ──────────────────────────────────────────────────────────
# server.py
# run with:  python server.py
# Requires:  pip install flask flask_cors
#
# IMPORTANT:
# • The file enhanced_chatbot_cohere_parallel_FIXED.py must be in the
#   same folder (it is imported here).
# • Inside that file scroll to the very bottom of
#   ask_parallel_enhanced_chatbot(...) and ADD  ➜  return final_answer
#   so the function actually gives us the answer string.
# ──────────────────────────────────────────────────────────
import json, uuid, time, threading
from collections import defaultdict
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from chatbot_agent_claw4 import ask_parallel_enhanced_chatbot, simplified_react_prompt_template
# top of file
from queue import Queue, Empty
import threading

app = Flask(__name__)
CORS(app, supports_credentials=True)

# ------------------------------------------------------------------
# in-memory chat log  ➜  {chat_id: [{"role": "..", "content": ".."}]}
# ------------------------------------------------------------------
CHAT_MEMORY = defaultdict(list)

def build_history_prompt(chat_id: str, max_pairs: int = 6) -> str:
    """Return the last <user, assistant> turns as markdown."""
    turns = CHAT_MEMORY.get(chat_id, [])[-max_pairs*2:]          # tail
    blocks = []
    for m in turns:
        who = "User" if m["role"] == "user" else "Assistant"
        blocks.append(f"**{who}:** {m['content']}")
    return "\n".join(blocks)

def new_chat_id() -> str:
    return str(uuid.uuid4())

# ---------------------------------------------------------------
# Helper to build SSE lines ( *always* ends with blank line "\n\n")
# ---------------------------------------------------------------
def sse_pack(payload: dict) -> str:
    return f"data:{json.dumps(payload, ensure_ascii=False)}\n\n"

# ------------------------------------------------------------------
#   /new_chat  – used by the UI when the user clicks “New chat”
# ------------------------------------------------------------------
@app.route("/new_chat")
def new_chat():
    cid = new_chat_id()
    return jsonify({"chat_id": cid})

# ------------------------------------------------------------------
#   /ask_stream  – main SSE endpoint
#   GET  /ask_stream?chat_id=xxx&query=some%20question
# ------------------------------------------------------------------
@app.route("/ask_stream")
def ask_stream():
    chat_id = request.args.get("chat_id") or new_chat_id()
    query   = request.args.get("query", "").strip()

    status_queue = Queue()          # we could use this later for more granularity
    token_queue  = Queue()          # every new token from the LLM lands here

    # ---------------- worker that runs the agent ------------------
    def worker():
        # 0) record the user question immediately  🔹
        CHAT_MEMORY[chat_id].append({"role": "user", "content": query})

        status_queue.put("thinking")
        status_queue.put("searching")

        # 1) build history (now contains the fresh user line as well)
        history_block = build_history_prompt(chat_id)

        contextual_query = (
            "🧠 Conversation so far (for your reference only, do NOT quote):\n"
            f"{history_block}\n\n"
            "### New User Question:\n"
            f"{query}"
        )

        # 2) run the agent
        assistant_answer = ask_parallel_enhanced_chatbot(
            user_query   = contextual_query,
            service      = "openrouter",
            model_name   = "mistralai/mistral-small-3.2-24b-instruct",
            stream_answer=True,
            token_queue  = token_queue,
        )

        # 3) store the assistant reply  🔹
        CHAT_MEMORY[chat_id].append({"role": "assistant", "content": assistant_answer})

        token_queue.put(None)
        status_queue.put("finished")

    threading.Thread(target=worker, daemon=True).start()

    # ---------------- generator that feeds the browser ------------
    def stream():
        # 1) initial status
        yield sse_pack({"type": "status", "status": "thinking"})

        # Pump queues
        answer_started = False
        while True:
            # first, check token queue without blocking too long
            try:
                token = token_queue.get(timeout=0.05)
                if token is None:                       # sentinel → finished
                    break
                # first token ⇒ tell UI we’re answering
                if not answer_started:
                    yield sse_pack({"type": "status", "status": "answering"})
                    answer_started = True
                yield sse_pack({"type": "answer", "chunk": token})
            except Empty:
                pass

            # also relay any pending status messages
            while not status_queue.empty():
                stat = status_queue.get_nowait()
                yield sse_pack({"type": "status", "status": stat})

        # final “finished”
        yield sse_pack({"type": "status", "status": "finished"})

    # expose chat_id for first request
    headers = {"X-Chat-Id": chat_id}
    return Response(stream(), headers=headers, mimetype="text/event-stream")

@app.route("/ask", methods=["POST"])
def ask():
    data     = request.get_json(force=True)
    query    = data.get("query", "")
    chat_id  = data.get("chat_id") or new_chat_id()
    answer   = ask_parallel_enhanced_chatbot(
        user_query=query,
        service      = "openrouter",
        model_name   = "mistralai/mistral-small-3.2-24b-instruct",
        stream_answer= False
    )
    # memory
    CHAT_MEMORY[chat_id].append({"role": "user"     , "content": query })
    CHAT_MEMORY[chat_id].append({"role": "assistant", "content": answer})
    return jsonify({"chat_id": chat_id, "answer": answer})

if __name__ == "__main__":
    # threaded=True → multiple concurrent EventSource connections
    app.run(host="0.0.0.0", port=5000, threaded=True)