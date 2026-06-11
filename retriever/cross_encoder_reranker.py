from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Cross-encoder reranker for IPL RAG system"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            self.model = CrossEncoder(self.model_name)
        except Exception as e:
            print(f"Warning: Failed to load cross-encoder model: {e}")
            self.model = None
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, rerank_score) tuples sorted by relevance
        """
        if not self.model or not documents:
            # Return documents with default scores if model not available
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        # Prepare query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Sort documents by scores (descending)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return doc_score_pairs[:top_k]
    
    def rerank_with_scores(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 4
    ) -> Tuple[List[Document], List[float]]:
        """
        Rerank documents and return separate lists of documents and scores.
        
        Args:
            query: Query string
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            Tuple of (documents list, scores list)
        """
        reranked = self.rerank(query, documents, top_k)
        
        if not reranked:
            return [], []
        
        documents = [doc for doc, score in reranked]
        scores = [score for doc, score in reranked]
        
        return documents, scores
