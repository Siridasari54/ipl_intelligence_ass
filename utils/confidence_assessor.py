from typing import List


def assess_confidence(
    retrieval_scores: List[float],
    rerank_scores: List[float],
    retrieval_threshold: float = 0.3,
    rerank_threshold: float = 0.5
) -> str:
    """
    Assess confidence level based on retrieval and reranking scores.
    
    Args:
        retrieval_scores: List of semantic retrieval scores (from Chroma)
        rerank_scores: List of reranking scores (from cross-encoder)
        retrieval_threshold: Threshold for considering retrieval scores as high
        rerank_threshold: Threshold for considering rerank scores as high
        
    Returns:
        Confidence level: "high", "medium", or "low"
    """
    if not retrieval_scores or not rerank_scores:
        return "low"
    
    # Calculate average scores
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores)
    avg_rerank = sum(rerank_scores) / len(rerank_scores)
    
    # Assess confidence based on both scores
    if avg_retrieval >= retrieval_threshold and avg_rerank >= rerank_threshold:
        return "high"
    elif avg_retrieval >= retrieval_threshold * 0.7 and avg_rerank >= rerank_threshold * 0.7:
        return "medium"
    else:
        return "low"


def get_confidence_message(confidence_level: str) -> str:
    """
    Get a user-friendly message based on confidence level.
    
    Args:
        confidence_level: Confidence level ("high", "medium", or "low")
        
    Returns:
        Message string
    """
    messages = {
        "high": "",
        "medium": "Note: The answer is based on moderately relevant information from the dataset.",
        "low": "I don't have enough reliable information from the indexed IPL dataset to confidently answer this question."
    }
    return messages.get(confidence_level, "")
