"""
30 recruiter-style test queries for Ask Manideep portfolio assistant.
Run: python test_queries.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from query_engine import process_query

TEST_QUERIES = [
    # Profile & Background
    ("Tell me about yourself", {}),
    ("What is your educational background?", {}),
    ("What university do you attend?", {}),
    ("What is your CGPA?", {}),
    ("What are your career goals?", {}),

]


async def run_tests():
    print("=" * 60)
    print("ASK MANIDEEP — Test Suite (30 Recruiter Queries)")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, (query, ctx) in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i:02d}] Query: {query}")
        if ctx:
            print(f"      Context: {ctx}")
        try:
            result = await process_query(query, ctx if ctx else None)
            answer_preview = result["answer"][:150].replace("\n", " ")
            print(f"      Sources: {result['context_used']}")
            print(f"      Answer:  {answer_preview}...")
            passed += 1
        except Exception as e:
            print(f"      ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed / {failed} failed out of {len(TEST_QUERIES)} queries")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_tests())
