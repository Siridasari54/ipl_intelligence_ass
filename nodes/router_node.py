import os
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router node to classify query type.
    
    Classifies queries into:
    - Team queries
    - Batting queries
    - Bowling queries
    - Venue queries
    - Records queries
    - Head-to-head (h2h) queries
    - Form queries
    - General IPL queries
    """
    question = state["question"]
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Create routing prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a query classifier for IPL (Indian Premier League) cricket information.
        Classify the user's question into one of these categories:
        
        1. team - Questions about IPL teams, franchises, squad, owners, team composition, etc.
        2. batting - Questions about batting statistics, runs, centuries, batting averages, strike rates, etc.
        3. bowling - Questions about bowling statistics, wickets, economy rates, bowling averages, etc.
        4. venue - Questions about stadiums, venues, locations, grounds, pitch conditions, etc.
        5. records - Questions about records, milestones, achievements, history, highest scores, etc.
        6. h2h - Questions about head-to-head matchups between teams, historical matchups, team vs team performance, etc.
        7. form - Questions about recent form, current performance trends, recent matches, current season performance, etc.
        8. general - General IPL questions that don't fit other categories
        
        Return ONLY the category name (lowercase, single word).
        
        Question: {question}
        
        Category:"""
    )
    
    # Classify query
    chain = prompt | llm
    response = chain.invoke({"question": question})
    query_type = response.content.strip().lower()
    
    # Validate query type
    valid_types = ["team", "batting", "bowling", "venue", "records", "h2h", "form", "general"]
    if query_type not in valid_types:
        query_type = "general"  # Default to general if classification fails
    
    return {
        "question": question,
        "query_type": query_type
    }
