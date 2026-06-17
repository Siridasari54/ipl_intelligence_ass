from typing import Dict, Any, List
from langchain_core.documents import Document


def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validation node to check if sufficient context was retrieved.
    
    Validates:
    - Whether documents were retrieved
    - Whether sufficient context exists
    - Whether answer generation should continue
    
    If validation fails, returns a message indicating insufficient information.
    """
    question = state["question"]
    query_type = state["query_type"]
    documents: List[Document] = state["documents"]
    retrieval_scores: List[float] = state["retrieval_scores"]
    
    # Check if documents were retrieved
    if not documents or len(documents) == 0:
        validation_status = "failed"
        generation = "I don't have enough information from the indexed IPL dataset to answer this question."
        return {
            "question": question,
            "query_type": query_type,
            "documents": documents,
            "retrieval_scores": retrieval_scores,
            "validation_status": validation_status,
            "generation": generation,
            "sources": []
        }
    
    # Check if retrieval scores indicate low relevance (threshold: 0.3)
    # Lower scores mean less relevant documents
    avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
    if avg_score < 0.3:
        validation_status = "failed"
        generation = "I don't have enough information from the indexed IPL dataset to answer this question."
        return {
            "question": question,
            "query_type": query_type,
            "documents": documents,
            "retrieval_scores": retrieval_scores,
            "validation_status": validation_status,
            "generation": generation,
            "sources": []
        }
    
    # Check if document content is sufficient (at least 50 characters total)
    total_content_length = sum(len(doc.page_content) for doc in documents)
    if total_content_length < 50:
        validation_status = "failed"
        generation = "I don't have enough information from the indexed IPL dataset to answer this question."
        return {
            "question": question,
            "query_type": query_type,
            "documents": documents,
            "retrieval_scores": retrieval_scores,
            "validation_status": validation_status,
            "generation": generation,
            "sources": []
        }
    
    # Validation passed
    validation_status = "passed"
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": retrieval_scores,
        "validation_status": validation_status
    }
