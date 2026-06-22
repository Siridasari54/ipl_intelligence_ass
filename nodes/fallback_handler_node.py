from typing import Dict, Any


def fallback_handler_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback handler node to provide safe responses when retrieval fails.
    
    Triggers when:
    - Retrieval returns empty results
    - Confidence is low
    - No relevant documents found
    - Grounding check fails
    
    Returns safe fallback response to prevent hallucinated answers.
    """
    question = state["question"]
    validation_status = state.get("validation_status", "")
    confidence_level = state.get("confidence_level", "")
    grounding_status = state.get("grounding_status", "")
    documents = state.get("documents", [])
    
    # Determine fallback reason
    fallback_reason = ""
    
    if not documents or len(documents) == 0:
        fallback_reason = "no_documents"
    elif validation_status == "failed":
        fallback_reason = "validation_failed"
    elif confidence_level == "low":
        fallback_reason = "low_confidence"
    elif grounding_status == "failed":
        fallback_reason = "grounding_failed"
    else:
        # No fallback needed
        return {
            "question": question,
            "fallback_triggered": False,
            "fallback_reason": None
        }
    
    # Generate safe fallback response
    fallback_responses = {
        "no_documents": "Data not available in IPL dataset. No relevant documents were found for this query.",
        "validation_failed": "Data not available in IPL dataset. The retrieved information was insufficient to answer this question.",
        "low_confidence": "Data not available in IPL dataset. I don't have enough confidence in the retrieved information to provide an accurate answer.",
        "grounding_failed": "Data not available in IPL dataset. The question requires information that is not present in the retrieved context."
    }
    
    fallback_message = fallback_responses.get(fallback_reason, "Data not available in IPL dataset.")
    
    return {
        "question": question,
        "fallback_triggered": True,
        "fallback_reason": fallback_reason,
        "generation": fallback_message,
        "sources": []
    }
