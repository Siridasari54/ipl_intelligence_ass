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
        """Split text into chunks"""
        chunks = self.text_splitter.split_text(text)
        documents = [
            Document(page_content=chunk, metadata={"source": source})
            for chunk in chunks
        ]
        return documents
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF file and return document chunks"""
        text = self.load_pdf(pdf_path)
        documents = self.split_text(text, source=pdf_path)
        return documents
