import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import logfire
from httpx import AsyncClient
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client as SupabaseClient

# Supabase Setup
SUPABASE_URL = "your-supabase-url"
SUPABASE_KEY = "your-supabase-key"
supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_KEY)

# Data Models
class CareTask(BaseModel):
    id: str
    title: str
    scheduled_time: str
    user_email: EmailStr
    reminder_sent: bool = False

class HealthReport(BaseModel):
    timestamp: str
    metrics: Dict[str, Any]  # e.g., {"sleep_quality": 4, "mood": "calm"}
    notes: Optional[str] = None

class MicrolearningModule(BaseModel):
    title: str
    summary: str
    tips: List[str]
    sources: List[str]

@dementia_care_agent.tool
async def schedule_care_task(
    ctx: RunContext[DementiaCareDeps],
    title: str,
    scheduled_time: str,
    user_email: EmailStr,
    reminder_minutes: int = 10,
) -> Dict[str, Any]:
    """Schedule a task (medication, meals, etc.) with Supabase and email reminders."""
    try:
        task = CareTask(
            id=f"task_{datetime.now().timestamp()}",
            title=title,
            scheduled_time=scheduled_time,
            user_email=user_email,
        )

        # Store in Supabase
        supabase.table("care_tasks").insert(task.model_dump()).execute()

        # Schedule email reminder
        reminder_time = datetime.fromisoformat(scheduled_time) - timedelta(minutes=reminder_minutes)
        await run_python(
            "send_email_reminder",
            args={"task": task.model_dump()},
            schedule_at=reminder_time,
        )

        return {"status": "scheduled", "task": task.model_dump()}
    except Exception as e:
        logfire.error(f"Task scheduling failed: {e}")
        raise ModelRetry("Could not schedule task. Please try again.")

@dementia_care_agent.tool
async def record_health_metric(
    ctx: RunContext[DementiaCareDeps],
    metric_type: str,
    value: Any,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """Record health data (sleep, mood, etc.) in Supabase."""
    try:
        report = HealthReport(
            timestamp=datetime.now().isoformat(),
            metrics={metric_type: value},
            notes=notes,
        )

        supabase.table("health_reports").insert(report.model_dump()).execute()
        return {"status": "recorded", "report": report.model_dump()}
    except Exception as e:
        logfire.error(f"Health metric recording failed: {e}")
        raise ModelRetry("Could not record metric. Please try again.")

@dementia_care_agent.tool
async def generate_weekly_report(
    ctx: RunContext[DementiaCareDeps],
) -> Dict[str, Any]:
    """Generate a weekly health summary from Supabase."""
    try:
        reports = supabase.table("health_reports").select("*").eq("user_id", ctx.deps.user_id).execute()
        return {
            "report": [
                {
                    "timestamp": r["timestamp"],
                    "metrics": r["metrics"],
                    "notes": r["notes"],
                }
                for r in reports.data
            ]
        }
    except Exception as e:
        logfire.error(f"Report generation failed: {e}")
        raise ModelRetry("Could not generate report. Please try again.")

@dementia_care_agent.tool
async def retrieve_emergency_guide(
    ctx: RunContext[DementiaCareDeps],
    situation: str,
) -> Dict[str, Any]:
    """Fetch step-by-step emergency protocols using RAG."""
    try:
        rag_results = await ctx.deps.rag_client.query(
            query=f"emergency protocol for {situation}",
            context="Caregiver needs immediate steps",
        )
        return {
            "steps": rag_results["content"].split("\n"),  # Convert to numbered list
            "sources": rag_results["sources"],
        }
    except Exception as e:
        logfire.error(f"Emergency guide retrieval failed: {e}")
        raise ModelRetry("Could not retrieve guide. Please try again.")

@dementia_care_agent.tool
async def deliver_microlearning(
    ctx: RunContext[DementiaCareDeps],
    topic: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch and deliver a microlearning module."""
    try:
        if not topic:
            topic = "communication strategies"  # Default or suggest based on history

        rag_results = await ctx.deps.rag_client.query(
            query=f"microlearning on {topic}",
            context="Caregiver needs concise tips",
        )

        module = MicrolearningModule(
            title=f"Quick Guide: {topic}",
            summary=rag_results["summary"],
            tips=rag_results["key_points"],
            sources=rag_results["sources"],
        )
        return {"module": module.model_dump()}
    except Exception as e:
        logfire.error(f"Microlearning failed: {e}")
        raise ModelRetry("Could not prepare module. Please try again.")

@dementia_care_agent.tool
async def emotional_check_in(
    ctx: RunContext[DementiaCareDeps],
    prompt: str = "How are you feeling today?",
) -> Dict[str, Any]:
    """Initiate an emotional check-in and log response."""
    try:
        supabase.table("check_ins").insert({
            "user_id": ctx.deps.user_id,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
        }).execute()
        return {"status": "check_in_initiated"}
    except Exception as e:
        logfire.error(f"Check-in failed: {e}")
        raise ModelRetry("Could not initiate check-in. Please try again.")

# Helper: Send email reminders (background task)
async def send_email_reminder(task: Dict[str, Any]):
    """Send email reminder via SendGrid or similar."""
    email_content = f"""
    Reminder: {task['title']} is scheduled for {task['scheduled_time']}.
    """
    await ctx.deps.email_client.send_email(
        to=task["user_email"],
        subject="Dementia Care Reminder",
        content=email_content,
    )