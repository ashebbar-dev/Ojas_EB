from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import logfire
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.mcp import run_python

@dataclass
class DementiaCareDeps:
    supabase_client: Any  # Supabase client
    rag_client: Any  # RAG service client
    email_client: Any  # For email reminders (e.g., SendGrid)
    user_id: str  # Current caregiver/user ID

dementia_care_agent = Agent(
    "anthropic:claude-3-opus",  # Or "openai:gpt-4o"
    system_prompt="""
        You are a compassionate dementia care assistant for caregivers. Provide:
        1. Accurate, cited answers using RAG.
        2. Gentle reminders for care tasks (medications, meals, appointments).
        3. Summarized health reports for doctors.
        4. Step-by-step emergency guides.
        5. Bite-sized microlearning modules.
        6. Emotional validation and support.

        Always use a warm, reassuring tone.
    """,
    deps_type=DementiaCareDeps,
    retries=2,
)

