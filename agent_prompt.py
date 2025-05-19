from typing import Dict

# System prompt extensions
EMERGENCY_PROMPT = """
In emergencies:
- Respond immediately with numbered steps.
- Cite sources (e.g., "CDC Guidelines suggest...").
- Use calm, directive language.
"""

MICROLEARNING_PROMPT = """
For microlearning:
- Title: "Quick Guide: [Topic]"
- 1-sentence intro.
- 3-5 bullet points.
- End with: "Reflect: [Question]".
"""

EMOTIONAL_PROMPT = """
For check-ins:
- Validate first ("I hear you...").
- Normalize feelings ("Many caregivers feel...").
- Offer 1-2 support resources.
"""

# Response templates
TEMPLATES = {
    "reminder": "⏰ I’ve scheduled a reminder for {task} at {time}. You’ll receive an email 10 minutes before.",
    "emergency": "🚨 For {situation}:\n{steps}\nSources: {sources}",
    "microlearning": "📚 {title}\n\n{summary}\n\nKey Tips:\n{tips}\n\nReflect: {question}",
    "emotional": "💙 {validation}\n{'Resources: ' + resources if resources else 'You’re doing great.'}",
}

def apply_template(template_name: str, data: Dict) -> str:
    return TEMPLATES[template_name].format(**data)


