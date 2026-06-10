from typing import List, Optional
from langchain_core.documents import Document
from vectorstore.chroma_store import ChromaVectorStore


class BaseRetriever:
    """Base retriever for IPL RAG system"""
    
    def __init__(self, vectorstore: ChromaVectorStore):
        self.vectorstore = vectorstore
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def retrieve_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """Retrieve relevant documents with similarity scores"""
        return self.vectorstore.similarity_search_with_score(query, k=k)
