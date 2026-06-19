import json
from typing import Dict, Any, List
from langchain_groq import ChatGroq


def query_decomposition_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Query decomposition node to break complex queries into sub-queries.
    
    Takes the rewritten query and decomposes it into 2-5 sub-queries
    for multi-hop retrieval and parallel search across different IPL data sections.
    
    Input: rewritten_query
    Output: sub_queries in JSON list format
    """
    rewritten_query = state["rewritten_query"]
    question = state["question"]
    
    # Initialize LLM for query decomposition
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    # System prompt for query decomposition
    system_prompt = """You are an IPL cricket query decomposition expert. Your task is to break down complex IPL queries into 2-5 simpler sub-queries for parallel retrieval.

Rules:
1. Decompose complex queries into 2-5 sub-queries
2. Each sub-query should focus on a specific aspect (batting, bowling, venue, H2H, team, form, records)
3. Sub-queries should be independent and can be executed in parallel
4. Output MUST be valid JSON format: ["sub_query_1", "sub_query_2", ...]
5. Do not answer the questions - only decompose them
6. Keep sub-queries concise and retrieval-focused
7. If the query is simple, return it as a single-item list

Examples:
- "Compare Kohli and Rohit's batting stats" → ["Virat Kohli batting statistics IPL", "Rohit Sharma batting statistics IPL"]
- "CSK vs MI at Chepauk" → ["Chennai Super Kings vs Mumbai Indians head to head", "MA Chidambaram Stadium pitch conditions"]
- "Best bowler in IPL history" → ["Highest wicket taker IPL history", "Best economy rate IPL history", "Best bowling average IPL history"]
- "RCB team composition 2024" → ["Royal Challengers Bangalore squad 2024", "RCB batting lineup 2024", "RCB bowling attack 2024"]

Decompose the following query into 2-5 sub-queries and return ONLY the JSON array:"""

    # Generate sub-queries
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rewritten_query}
    ])
    
    # Parse JSON response
    try:
        sub_queries = json.loads(response.content.strip())
        if not isinstance(sub_queries, list):
            sub_queries = [rewritten_query]
    except json.JSONDecodeError:
        # Fallback to single query if JSON parsing fails
        sub_queries = [rewritten_query]
    
    return {
        "question": question,
        "rewritten_query": rewritten_query,
        "sub_queries": sub_queries
    }
