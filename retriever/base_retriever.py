from typing import List, Optional
from langchain_core.documents import Document
from vectorstore.chroma_store import ChromaVectorStore


class BaseRetriever:
    """Base retriever for IPL RAG system"""
    
    def __init__(self, vectorstore: ChromaVectorStore):
        self.vectorstore = vectorstore
        self.all_documents: Optional[List[Document]] = None
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def retrieve_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """Retrieve relevant documents with similarity scores"""
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def load_all_documents(self) -> List[Document]:
        """Load all documents from vectorstore for BM25 indexing"""
        if self.all_documents is None:
            if self.vectorstore.vectorstore is None:
                self.vectorstore.load_vectorstore()
            # Get all documents from the collection
            self.all_documents = self.vectorstore.vectorstore.get()
            # Convert to Document objects
            self.all_documents = [
                Document(page_content=doc, metadata=metadata)
                for doc, metadata in zip(self.all_documents['documents'], self.all_documents['metadatas'])
            ]
        return self.all_documents
    
    def retrieve_by_metadata(self, metadata_section: str) -> List[Document]:
        """Retrieve documents filtered by metadata section"""
        all_docs = self.load_all_documents()
        filtered = [
            doc for doc in all_docs
            if doc.metadata.get("metadata_section") == metadata_section
        ]
        return filtered
