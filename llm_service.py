import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are Manideep's AI portfolio assistant — a smart, concise representative of Alur Manideep, a Computer Science graduate (VIT-AP University, 2026) specializing in AI/ML, Computer Vision, NLP, Android Development, and Automation.

Your job is to answer questions from recruiters and visitors about Manideep's background, projects, skills, certifications, and achievements based strictly on the provided knowledge base context.

Rules:
- Answer confidently as Manideep's representative
- Be professional but conversational
- Be specific — cite project names, technologies, metrics
- Keep answers focused and well-structured (use bullet points for lists)
- If asked about something not in the context, say you don't have that info but offer what you do know
- Never fabricate information not present in the context
- For project questions: always mention the technologies, key achievements, and what problem it solved
- Keep responses readable — not too short, not too long
- Answer in FIRST PERSON as if Manideep himself is speaking
"""


def build_user_prompt(query: str, context: dict, conversation_context: dict = None) -> str:
    parts = [f"User question: {query}\n"]

    if conversation_context:
        parts.append(
            f"Conversation context: {json.dumps(conversation_context)}\n"
        )

    parts.append("Relevant knowledge base context:\n")
    parts.append(json.dumps(context, indent=2))

    parts.append(
        "\nPlease answer the question based only on the above context."
    )

    return "\n".join(parts)


async def call_llm(
    query: str,
    context: dict,
    conversation_context: dict = None
) -> str:

    if not GROQ_API_KEY:
        return (
            "⚠️ GROQ_API_KEY not configured. "
            "Please set GROQ_API_KEY in your .env file."
        )

    try:
        user_prompt = build_user_prompt(
            query,
            context,
            conversation_context
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Groq API Error: {str(e)}"