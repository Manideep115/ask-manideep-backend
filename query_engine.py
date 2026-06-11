from retriever import retrieve_context
from llm_service import call_llm


async def process_query(query: str, conversation_context: dict = None) -> dict:
    """
    Main query processing pipeline:
    1. Retrieve relevant context from knowledge base
    2. Single LLM call to generate answer
    """
    # Step 1: Retrieve context
    retrieval_result = retrieve_context(query, conversation_context)

    # Step 2: Single LLM call
    answer = await call_llm(
        query=query,
        context=retrieval_result["context"],
        conversation_context=conversation_context
    )

    return {
        "answer": answer,
        "context_used": retrieval_result["sources"],
        "intents": retrieval_result["intents"],
        "top_project": retrieval_result.get("top_project")
    }
