from typing import Dict, Any, List
from langchain_core.documents import Document
from retriever.base_retriever import BaseRetriever


def retrieval_node(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Retrieval node to fetch relevant documents from vector store.
    
    Retrieves the most relevant chunks based on the query and query type.
    """
    question = state["question"]
    query_type = state["query_type"]
    
    # Retrieve documents with scores
    documents_with_scores = retriever.retrieve_with_scores(question, k=4)
    
    # Extract documents and scores
    documents = [doc for doc, score in documents_with_scores]
    scores = [score for doc, score in documents_with_scores]
    
    # Add metadata to documents
    for i, doc in enumerate(documents):
        doc.metadata["retrieval_score"] = scores[i]
        doc.metadata["query_type"] = query_type
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": scores
    }
