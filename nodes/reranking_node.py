from typing import Dict, Any, List
from langchain_core.documents import Document
from retriever.cross_encoder_reranker import CrossEncoderReranker


def reranking_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reranking node to rerank retrieved documents using cross-encoder.
    
    Takes the top 10 candidate documents from hybrid retrieval and reranks them
    using cross-encoder to select the best top 4 for answer generation.
    """
    question = state["question"]
    query_type = state["query_type"]
    documents: List[Document] = state["documents"]
    retrieval_scores: List[float] = state["retrieval_scores"]
    
    # Initialize cross-encoder reranker
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Rerank documents
    reranked_docs, rerank_scores = reranker.rerank_with_scores(
        query=question,
        documents=documents,
        top_k=4
    )
    
    # Update metadata with rerank scores
    for i, doc in enumerate(reranked_docs):
        doc.metadata["rerank_score"] = rerank_scores[i]
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": reranked_docs,
        "retrieval_scores": retrieval_scores,
        "rerank_scores": rerank_scores
    }
