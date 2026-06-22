from typing import Dict, Any, List
from utils.confidence_assessor import assess_confidence, get_confidence_message


def confidence_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Confidence assessment node to determine confidence level based on retrieval and reranking scores.
    
    Uses retrieval scores and reranking scores to determine confidence levels:
    - High confidence
    - Medium confidence
    - Low confidence
    
    If confidence is low, returns a response indicating insufficient reliable information.
    """
    question = state["question"]
    query_type = state["query_type"]
    documents: List = state["documents"]
    retrieval_scores: List[float] = state["retrieval_scores"]
    rerank_scores: List[float] = state["rerank_scores"]
    
    # Assess confidence level
    confidence_level = assess_confidence(retrieval_scores, rerank_scores)
    
    # If confidence is low, return early with error message
    if confidence_level == "low":
        confidence_message = get_confidence_message(confidence_level)
        return {
            "question": question,
            "query_type": query_type,
            "documents": documents,
            "retrieval_scores": retrieval_scores,
            "rerank_scores": rerank_scores,
            "confidence_level": confidence_level,
            "generation": confidence_message,
            "sources": []
        }
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": retrieval_scores,
        "rerank_scores": rerank_scores,
        "confidence_level": confidence_level
    }
