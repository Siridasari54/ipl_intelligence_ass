from typing import Dict, Any
from langchain_groq import ChatGroq


def query_rewrite_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query rewriting node to optimize user queries for IPL retrieval.
    
    Takes the original user query and rewrites it to be more effective
    for retrieval without answering the question. Focuses on:
    - Adding IPL-specific terminology
    - Expanding abbreviations
    - Clarifying ambiguous terms
    - Structuring for better retrieval
    
    Input: user query
    Output: rewritten_query optimized for IPL retrieval
    """
    question = state["question"]
    
    # Initialize LLM for query rewriting
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    # System prompt for query rewriting
    system_prompt = """You are an IPL cricket query optimization expert. Your task is to rewrite user queries to make them more effective for retrieval from an IPL database.

Rules:
1. DO NOT answer the question - only rewrite it for better retrieval
2. Add IPL-specific terminology (e.g., "strike rate" instead of "SR", "economy rate" instead of "econ")
3. Expand abbreviations (e.g., "CSK" to "Chennai Super Kings", "MI" to "Mumbai Indians")
4. Clarify ambiguous terms (e.g., "recent form" to "2024 IPL season form")
5. Add relevant context for better retrieval (e.g., "batting statistics" for batting queries)
6. Keep the query concise and focused
7. Maintain the original intent of the question

Examples:
- "Who has best SR?" → "Which player has the highest batting strike rate in IPL history?"
- "CSK vs MI record" → "Head-to-head record between Chennai Super Kings and Mumbai Indians"
- "Best economy" → "Which bowler has the best economy rate in IPL?"
- "Recent form of Kohli" → "Virat Kohli's batting form in the 2024 IPL season"

Rewrite the following query for optimal IPL retrieval:"""

    # Generate rewritten query
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ])
    
    rewritten_query = response.content.strip()
    
    return {
        "question": question,
        "rewritten_query": rewritten_query
    }
