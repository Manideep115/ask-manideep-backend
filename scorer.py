import json
import re
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


def load_aliases() -> dict:
    with open(KNOWLEDGE_DIR / "aliases.json") as f:
        return json.load(f)


def load_projects_index() -> dict:
    with open(KNOWLEDGE_DIR / "projects_index.json") as f:
        return json.load(f)


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "i", "you", "your", "yours",
    "yourself", "me", "my", "mine", "we", "our", "ours", "they", "their",
    "it", "its", "this", "that", "these", "those", "what", "which", "who",
    "whom", "tell", "about", "and", "or", "but", "if", "of", "at", "by",
    "for", "with", "to", "from", "in", "on", "out", "up", "down", "can",
    "could", "would", "should", "will", "shall", "may", "might", "must",
    "not", "no", "yes", "any", "all", "some", "more", "most", "other",
    "such", "than", "too", "very", "just", "so", "only", "own", "same",
    "as", "into", "over", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "each", "few",
}


def normalize(text: str) -> str:
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))


def tokenize_filtered(text: str) -> list[str]:
    """Tokenize and remove stopwords + short tokens."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


def expand_query_terms(query: str, aliases: dict) -> list[str]:
    """Expand query with alias terms."""
    tokens = tokenize_filtered(query)
    expanded = set(tokens)
    query_lower = normalize(query)

    for domain, terms in aliases.items():
        # Check if domain key itself appears
        if domain in query_lower or domain.replace("_", " ") in query_lower:
            expanded.update(tokenize_filtered(" ".join(terms)))
        # Check if any alias term appears in query
        for term in terms:
            if term in query_lower:
                expanded.update(tokenize_filtered(term))
                expanded.update(tokenize_filtered(" ".join(terms)))
                break

    return list(expanded)


def score_project(project_id: str, project_meta: dict, query_terms: list[str], original_query: str) -> float:
    """Score a project against query terms."""
    score = 0.0
    query_lower = normalize(original_query)
    query_term_set = set(query_terms)

    # Exact project name match (highest weight)
    project_title = normalize(project_meta.get("title", project_id))
    if project_title in query_lower or project_id.replace("_", " ") in query_lower:
        score += 50.0

    # Alias matches
    aliases = project_meta.get("aliases", [])
    for alias in aliases:
        if normalize(alias) in query_lower:
            score += 40.0

    # Keyword matches (exact token match)
    for kw in project_meta.get("keywords", []):
        kw_tokens = set(tokenize_filtered(kw))
        if kw_tokens and kw_tokens.issubset(query_term_set):
            score += 6.0
        elif kw_tokens & query_term_set:
            score += 3.0

    # Domain matches (exact token match)
    for domain in project_meta.get("domains", []):
        domain_tokens = set(tokenize_filtered(domain))
        if domain_tokens & query_term_set:
            score += 8.0

    # Priority weight (base relevance) - small tiebreaker only
    priority = project_meta.get("priority", 5)
    score += priority * 0.3

    return score


def rank_projects(query: str, context: dict = None) -> list[dict]:
    """Rank all projects by relevance to query."""
    aliases = load_aliases()
    projects_index = load_projects_index()
    query_terms = expand_query_terms(query, aliases)

    scored = []
    for project_id, meta in projects_index.items():
        s = score_project(project_id, meta, query_terms, query)
        # Boost if context references this project
        if context and context.get("last_project") == project_id:
            s += 30.0
        scored.append({"id": project_id, "score": s, "meta": meta})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
