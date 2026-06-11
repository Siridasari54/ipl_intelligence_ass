import os
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Process PDF documents for IPL RAG system"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Keywords for metadata classification
        self.metadata_keywords = {
            "team": ["team", "franchise", "squad", "owner", "captain", "coach", "players", "roster"],
            "batting": ["bat", "run", "score", "century", "half-century", "strike rate", "batting average", "runs scored"],
            "bowling": ["bowl", "wicket", "economy", "bowling average", "maiden", "spell", "figures"],
            "venue": ["stadium", "ground", "venue", "pitch", "capacity", "location", "city"],
            "records": ["record", "highest", "most", "best", "milestone", "achievement", "history"],
            "h2h": ["vs", "against", "matchup", "head to head", "faced", "encounter", "rivalry"],
            "form": ["recent", "form", "last", "current season", "performance", "trend", "streak"]
        }
    
    def classify_chunk(self, chunk: str) -> str:
        """
        Classify a chunk into a metadata category based on keyword matching.
        Returns 'general' if no specific category matches.
        """
        chunk_lower = chunk.lower()
        scores = {}
        
        for category, keywords in self.metadata_keywords.items():
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            if score > 0:
                scores[category] = score
        
        if not scores:
            return "general"
        
        # Return category with highest score
        return max(scores, key=scores.get)
    
    def load_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    
    def split_text(self, text: str, source: str = "") -> List[Document]:
        """Split text into chunks with metadata classification"""
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            metadata_section = self.classify_chunk(chunk)
            metadata = {
                "source": source,
                "chunk_id": i,
                "metadata_section": metadata_section
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
        
        return documents
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF file and return document chunks with metadata"""
        text = self.load_pdf(pdf_path)
        documents = self.split_text(text, source=pdf_path)
        return documents
