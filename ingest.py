import os
from utils.document_processor import DocumentProcessor
from vectorstore.chroma_store import ChromaVectorStore
from dotenv import load_dotenv

load_dotenv()


def ingest_pdf(pdf_path: str):
    """Ingest PDF file and create vector store"""
    print(f"Processing PDF: {pdf_path}")
    
    # Initialize document processor
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    
    # Process PDF
    documents = processor.process_pdf(pdf_path)
    print(f"Created {len(documents)} document chunks")
    
    # Initialize vector store
    vector_store = ChromaVectorStore(persist_directory="./data/chroma_db")
    
    # Create vector store
    vector_store.create_vectorstore(documents)
    print("Vector store created successfully!")
    
    print(f"\nIngestion complete! You can now run the app with: streamlit run ui/streamlit_app.py")


if __name__ == "__main__":
    # Default PDF path - update this to your PDF location
    pdf_path = r"c:\Users\sirid\Downloads\IPL_LangGraph_RAG_Dataset.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        print("Please update the pdf_path variable in ingest.py to point to your PDF file.")
        exit(1)
    
    ingest_pdf(pdf_path)
