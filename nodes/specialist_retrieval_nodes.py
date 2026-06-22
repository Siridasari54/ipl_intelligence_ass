import os
import pickle
from typing import Dict, Any, List
from langchain_core.documents import Document
from retriever.base_retriever import BaseRetriever
from retriever.bm25_retriever import BM25Retriever
from utils.numeric_filter import detect_numeric_query, parse_numeric_filters, numeric_filter


def hybrid_retrieval(
    question: str, 
    metadata_section: str, 
    retriever: BaseRetriever,
    k: int = 10
) -> List[tuple]:
    """
    Perform hybrid retrieval using Chroma semantic search and BM25 keyword search.
    
    Args:
        question: Query string
        metadata_section: Metadata section to filter by
        retriever: Base retriever instance
        k: Number of results to retrieve from each method
        
    Returns:
        List of (document, score) tuples merged and deduplicated
    """
    # Check for numeric query and apply pandas filter first
    if detect_numeric_query(question):
        try:
            # Load DataFrame for numeric filtering
            if os.path.exists("./data/numeric_dataframe.pkl"):
                with open("./data/numeric_dataframe.pkl", "rb") as f:
                    dataframe = pickle.load(f)
                
                # Parse numeric filters from query
                filters = parse_numeric_filters(question, metadata_section)
                
                # Apply numeric filter
                numeric_results = numeric_filter(dataframe, metadata_section, filters)
                
                if numeric_results:
                    # Convert numeric results to Document objects
                    numeric_docs = [
                        Document(page_content=row, metadata={"section": metadata_section, "source": "numeric_filter"})
                        for row in numeric_results[:k]
                    ]
                    # Return with high scores to prioritize
                    return [(doc, 1.0) for doc in numeric_docs]
        except Exception as e:
            # Fall back to normal retrieval if numeric filter fails
            pass
    
    # Filter documents by metadata section
    filtered_docs = retriever.retrieve_by_metadata(metadata_section)
    
    if not filtered_docs:
        # If no filtered docs, fall back to general retrieval
        chroma_results = retriever.retrieve_with_scores(question, k=k)
        return chroma_results
    
    # Chroma semantic retrieval on filtered documents
    # Note: Chroma doesn't support filtering on the fly well, so we'll use general retrieval
    # and then filter results by metadata
    chroma_results = retriever.retrieve_with_scores(question, k=k*2)
    chroma_filtered = [
        (doc, score) for doc, score in chroma_results
        if doc.metadata.get("metadata_section") == metadata_section
    ]
    
    # BM25 retrieval on filtered documents
    bm25_retriever = BM25Retriever(filtered_docs)
    bm25_results = bm25_retriever.retrieve_with_scores(question, k=k)
    
    # Merge and deduplicate results
    seen_docs = set()
    merged_results = []
    
    # Add Chroma results
    for doc, score in chroma_filtered:
        doc_id = doc.metadata.get("chunk_id", id(doc))
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            merged_results.append((doc, score))
    
    # Add BM25 results (if not already seen)
    for doc, score in bm25_results:
        doc_id = doc.metadata.get("chunk_id", id(doc))
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            merged_results.append((doc, score))
    
    # Sort by score (descending) and return top-k
    merged_results.sort(key=lambda x: x[1], reverse=True)
    return merged_results[:k]


def team_retrieval_node(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Specialist retrieval node for team-related queries.
    
    Performs metadata filtering and hybrid retrieval (Chroma + BM25).
    """
    question = state["question"]
    query_type = state["query_type"]
    
    # Perform hybrid retrieval with metadata filtering
    documents_with_scores = hybrid_retrieval(question, "team", retriever, k=10)
    
    # Extract documents and scores
    documents = [doc for doc, score in documents_with_scores]
    scores = [score for doc, score in documents_with_scores]
    
    # Add metadata to documents
    for i, doc in enumerate(documents):
        doc.metadata["retrieval_score"] = scores[i]
        doc.metadata["query_type"] = query_type
        doc.metadata["specialist"] = "team"
        doc.metadata["metadata_section"] = "team"
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": scores
    }


def batting_retrieval_node(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Specialist retrieval node for batting-related queries.
    
    Performs metadata filtering and hybrid retrieval (Chroma + BM25).
    """
    question = state["question"]
    query_type = state["query_type"]
    
    # Perform hybrid retrieval with metadata filtering
    documents_with_scores = hybrid_retrieval(question, "batting", retriever, k=10)
    
    # Extract documents and scores
    documents = [doc for doc, score in documents_with_scores]
    scores = [score for doc, score in documents_with_scores]
    
    # Add metadata to documents
    for i, doc in enumerate(documents):
        doc.metadata["retrieval_score"] = scores[i]
        doc.metadata["query_type"] = query_type
        doc.metadata["specialist"] = "batting"
        doc.metadata["metadata_section"] = "batting"
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": scores
    }


def bowling_retrieval_node(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Specialist retrieval node for bowling-related queries.
    
    Performs metadata filtering and hybrid retrieval (Chroma + BM25).
    """
    question = state["question"]
    query_type = state["query_type"]
    
    # Perform hybrid retrieval with metadata filtering
    documents_with_scores = hybrid_retrieval(question, "bowling", retriever, k=10)
    
    # Extract documents and scores
    documents = [doc for doc, score in documents_with_scores]
    scores = [score for doc, score in documents_with_scores]
    
    # Add metadata to documents
    for i, doc in enumerate(documents):
        doc.metadata["retrieval_score"] = scores[i]
        doc.metadata["query_type"] = query_type
        doc.metadata["specialist"] = "bowling"
        doc.metadata["metadata_section"] = "bowling"
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": scores
    }


def venue_retrieval_node(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Specialist retrieval node for venue-related queries.
    
    Performs metadata filtering and hybrid retrieval (Chroma + BM25).
    """
    question = state["question"]
    query_type = state["query_type"]
    
    # Perform hybrid retrieval with metadata filtering
    documents_with_scores = hybrid_retrieval(question, "venue", retriever, k=10)
    
    # Extract documents and scores
    documents = [doc for doc, score in documents_with_scores]
    scores = [score for doc, score in documents_with_scores]
    
    # Add metadata to documents
    for i, doc in enumerate(documents):
        doc.metadata["retrieval_score"] = scores[i]
        doc.metadata["query_type"] = query_type
        doc.metadata["specialist"] = "venue"
        doc.metadata["metadata_section"] = "venue"
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": scores
    }


def records_retrieval_node(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Specialist retrieval node for records-related queries.
    
    Performs metadata filtering and hybrid retrieval (Chroma + BM25).
    """
    question = state["question"]
    query_type = state["query_type"]
    
    # Perform hybrid retrieval with metadata filtering
    documents_with_scores = hybrid_retrieval(question, "records", retriever, k=10)
    
    # Extract documents and scores
    documents = [doc for doc, score in documents_with_scores]
    scores = [score for doc, score in documents_with_scores]
    
    # Add metadata to documents
    for i, doc in enumerate(documents):
        doc.metadata["retrieval_score"] = scores[i]
        doc.metadata["query_type"] = query_type
        doc.metadata["specialist"] = "records"
        doc.metadata["metadata_section"] = "records"
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": scores
    }


def h2h_retrieval_node(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Specialist retrieval node for head-to-head (h2h) queries.
    
    Performs metadata filtering and hybrid retrieval (Chroma + BM25).
    """
    question = state["question"]
    query_type = state["query_type"]
    
    # Perform hybrid retrieval with metadata filtering
    documents_with_scores = hybrid_retrieval(question, "h2h", retriever, k=10)
    
    # Extract documents and scores
    documents = [doc for doc, score in documents_with_scores]
    scores = [score for doc, score in documents_with_scores]
    
    # Add metadata to documents
    for i, doc in enumerate(documents):
        doc.metadata["retrieval_score"] = scores[i]
        doc.metadata["query_type"] = query_type
        doc.metadata["specialist"] = "h2h"
        doc.metadata["metadata_section"] = "h2h"
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": scores
    }


def form_retrieval_node(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Specialist retrieval node for form-related queries.
    
    Performs metadata filtering and hybrid retrieval (Chroma + BM25).
    """
    question = state["question"]
    query_type = state["query_type"]
    
    # Perform hybrid retrieval with metadata filtering
    documents_with_scores = hybrid_retrieval(question, "form", retriever, k=10)
    
    # Extract documents and scores
    documents = [doc for doc, score in documents_with_scores]
    scores = [score for doc, score in documents_with_scores]
    
    # Add metadata to documents
    for i, doc in enumerate(documents):
        doc.metadata["retrieval_score"] = scores[i]
        doc.metadata["query_type"] = query_type
        doc.metadata["specialist"] = "form"
        doc.metadata["metadata_section"] = "form"
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": scores
    }


def general_retrieval_node(state: Dict[str, Any], retriever: BaseRetriever) -> Dict[str, Any]:
    """
    Specialist retrieval node for general IPL queries.
    
    Performs metadata filtering and hybrid retrieval (Chroma + BM25).
    """
    question = state["question"]
    query_type = state["query_type"]
    
    # Perform hybrid retrieval with metadata filtering
    documents_with_scores = hybrid_retrieval(question, "general", retriever, k=10)
    
    # Extract documents and scores
    documents = [doc for doc, score in documents_with_scores]
    scores = [score for doc, score in documents_with_scores]
    
    # Add metadata to documents
    for i, doc in enumerate(documents):
        doc.metadata["retrieval_score"] = scores[i]
        doc.metadata["query_type"] = query_type
        doc.metadata["specialist"] = "general"
        doc.metadata["metadata_section"] = "general"
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": scores
    }
