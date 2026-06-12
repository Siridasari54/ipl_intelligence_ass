from typing import List, Optional
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import re


class BM25Retriever:
    """BM25 keyword retriever for IPL RAG system"""
    
    def __init__(self, documents: Optional[List[Document]] = None):
        """
        Initialize BM25 retriever with documents.
        
        Args:
            documents: List of documents to index for BM25 retrieval
        """
        self.documents = documents if documents else []
        self.tokenized_corpus = []
        self.bm25 = None
        
        if self.documents:
            self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase, remove punctuation, split on whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return tokens
    
    def _build_index(self):
        """Build BM25 index from documents"""
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) 
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the BM25 index.
        
        Args:
            documents: List of documents to add
        """
        self.documents.extend(documents)
        self._build_index()
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve top-k documents using BM25.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        doc_scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_k_indices = sorted(
            range(len(doc_scores)), 
            key=lambda i: doc_scores[i], 
            reverse=True
        )[:k]
        
        # Return documents in order of relevance
        retrieved_docs = [self.documents[i] for i in top_k_indices]
        return retrieved_docs
    
    def retrieve_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """
        Retrieve top-k documents with BM25 scores.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        doc_scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_k_indices = sorted(
            range(len(doc_scores)), 
            key=lambda i: doc_scores[i], 
            reverse=True
        )[:k]
        
        # Return (document, score) tuples in order of relevance
        retrieved_with_scores = [
            (self.documents[i], doc_scores[i]) 
            for i in top_k_indices
        ]
        return retrieved_with_scores
    
    def filter_by_metadata(self, metadata_section: str) -> List[Document]:
        """
        Filter documents by metadata section.
        
        Args:
            metadata_section: Metadata section to filter by
            
        Returns:
            List of filtered documents
        """
        filtered = [
            doc for doc in self.documents 
            if doc.metadata.get("metadata_section") == metadata_section
        ]
        return filtered
