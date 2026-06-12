import json
import re
from pathlib import Path
from scorer import rank_projects, normalize

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
PROJECTS_DIR = KNOWLEDGE_DIR / "projects"

# Query intent categories
INTENT_PATTERNS = {
    "profile": [r"about you", r"who are you", r"introduce", r"tell me about yourself", r"background", r"summary"],
    "education": [r"education", r"degree", r"university", r"college", r"study", r"studied", r"gpa", r"cgpa", r"vit", r"school"],
    "skills": [r"skill", r"technology", r"tech stack", r"know", r"work with", r"experience with", r"familiar"],
    "certifications": [r"certif", r"course", r"oracle", r"cisco", r"credential"],
    "achievements": [r"achiev", r"award", r"recognition", r"honor", r"title", r"rank", r"won", r"win"],
    "contact": [r"contact", r"email", r"phone", r"reach", r"hire", r"connect"],
    "projects": [r"project", r"built", r"build", r"developed", r"created", r"work"],
    "all_projects": [r"all project", r"list project", r"what project", r"projects do you", r"projects have", r"projects worked on", r"project details", r"project info", r"project information",r"projects"],
    "best_projects": [r"best project", r"top \d*\s*project", r"favorite project",r"fav project", r"impressive", r"flagship", r"highlight", r"top projects"],
}


def detect_intents(query: str) -> list[str]:
    """Detect query intents."""
    query_lower = normalize(query)
    intents = []
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, query_lower):
                intents.append(intent)
                break
    return intents if intents else ["general"]


def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def load_project(project_id: str) -> dict:
    path = PROJECTS_DIR / f"{project_id}.json"
    if path.exists():
        return load_json(path)
    return {}


def retrieve_context(query: str, context: dict = None) -> dict:
    """Main retrieval function - returns structured context for LLM."""
    intents = detect_intents(query)
    retrieved = {}
    sources_used = []

    # Always include profile for general queries
    if "profile" in intents or "general" in intents or "about" in query.lower():
        retrieved["profile"] = load_json(KNOWLEDGE_DIR / "profile.json")
        sources_used.append("profile")

    if "education" in intents:
        retrieved["education"] = load_json(KNOWLEDGE_DIR / "education.json")
        sources_used.append("education")

    if "skills" in intents:
        retrieved["skills"] = load_json(KNOWLEDGE_DIR / "skills.json")
        sources_used.append("skills")

    if "certifications" in intents:
        retrieved["certifications"] = load_json(KNOWLEDGE_DIR / "certifications.json")
        sources_used.append("certifications")

    if "achievements" in intents:
        retrieved["achievements"] = load_json(KNOWLEDGE_DIR / "achievements.json")
        sources_used.append("achievements")

    if "contact" in intents:
        retrieved["contact"] = load_json(KNOWLEDGE_DIR / "contact.json")
        retrieved["links"] = load_json(KNOWLEDGE_DIR / "links.json")
        sources_used.append("contact")

    # Project retrieval
    ranked = rank_projects(query, context)

    if "all_projects" in intents:
        # Load top 5 projects with summaries only
        projects_data = []
        for proj in ranked[:5]:
            p = load_project(proj["id"])
            if p:
                projects_data.append({
                    "id": proj["id"],
                    "title": p.get("metadata", {}).get("title", proj["id"]),
                    "one_line": p.get("overview", {}).get("one_line", ""),
                    "domains": p.get("metadata", {}).get("domains", []),
                    "technologies": p.get("metadata", {}).get("technologies", []),
                    "timeline": p.get("metadata", {}).get("timeline", ""),
                })
        retrieved["projects_list"] = projects_data
        sources_used.append("all_projects")

    elif "best_projects" in intents:
        # Load top 3 projects with overviews
        projects_data = []
        for proj in ranked[:3]:
            p = load_project(proj["id"])
            if p:
                projects_data.append({
                    "id": proj["id"],
                    "title": p.get("metadata", {}).get("title", proj["id"]),
                    "one_line": p.get("overview", {}).get("one_line", ""),
                    "summary": p.get("overview", {}).get("summary", ""),
                    "technologies": p.get("metadata", {}).get("technologies", []),
                    "priority": proj["meta"].get("priority", 0),
                })
        retrieved["best_projects"] = projects_data
        sources_used.append("best_projects")

    elif ranked and ranked[0]["score"] > 10:
        # Load single most relevant project in full
        top = ranked[0]
        p = load_project(top["id"])
        if p:
            retrieved["project"] = p
            sources_used.append(f"project:{top['id']}")

    # Context-aware: if conversation references a project
    if context and context.get("last_project"):
        pid = context["last_project"]
        if f"project:{pid}" not in sources_used:
            p = load_project(pid)
            if p and not retrieved.get("project"):
                retrieved["project"] = p
                sources_used.append(f"project:{pid}_context")

    # For general intro queries, include brief projects overview
    if "profile" in intents and "projects" not in retrieved and "project" not in retrieved:
        index = load_json(KNOWLEDGE_DIR / "projects_index.json")
        brief = []
        for pid, meta in sorted(index.items(), key=lambda x: x[1].get("priority", 0), reverse=True)[:3]:
            brief.append({"id": pid, "domains": meta.get("domains", [])})
        retrieved["top_project_ids"] = brief
        sources_used.append("projects_overview")

    return {
        "context": retrieved,
        "sources": sources_used,
        "intents": intents,
        "top_project": ranked[0]["id"] if ranked else None
    }
