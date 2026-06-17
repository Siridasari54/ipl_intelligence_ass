import os
import re
import pickle
import pandas as pd
from typing import List, Dict, Any
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
        
        # Store DataFrame for numeric filtering
        self.dataframe = None
    
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
    
    def parse_table_row(self, row_text: str, section: str) -> Dict[str, Any]:
        """Parse a table row and extract metadata based on section type"""
        metadata = {"section": section, "source": "primary"}
        
        if section == "batting":
            # Extract player name, team, role from batting row
            parts = row_text.split("|")
            if len(parts) >= 3:
                metadata["player_name"] = parts[0].strip()
                metadata["team"] = parts[1].strip()
                role_map = {"Opener": "Opener", "Middle": "Middle-order", "WK": "WK-Bat", 
                           "AR": "All-rounder", "Finisher": "WK-Finisher"}
                metadata["role"] = role_map.get(parts[2].strip(), "Middle-order")
        
        elif section == "bowling":
            # Extract player name, team, bowl type from bowling row
            parts = row_text.split("|")
            if len(parts) >= 3:
                metadata["player_name"] = parts[0].strip()
                metadata["team"] = parts[1].strip()
                bowl_types = ["Leg-spin", "Off-spin", "Pace", "Medium-fast", "Mystery-spin"]
                metadata["bowl_type"] = parts[2].strip() if parts[2].strip() in bowl_types else "Pace"
        
        elif section == "form":
            # Extract player name, team from form row
            parts = row_text.split("|")
            if len(parts) >= 2:
                metadata["player_name"] = parts[0].strip()
                metadata["team"] = parts[1].strip()
                metadata["season"] = "2024"
        
        elif section == "venue":
            # Extract venue name, city, pitch type from venue row
            parts = row_text.split("|")
            if len(parts) >= 3:
                metadata["venue_name"] = parts[0].strip()
                metadata["city"] = parts[1].strip()
                pitch_types = ["slow", "flat", "bouncy", "balanced"]
                metadata["pitch_type"] = parts[2].strip() if parts[2].strip() in pitch_types else "balanced"
        
        elif section == "h2h":
            # Extract team1, team2 from H2H row - store twice for bidirectional lookup
            parts = row_text.split("|")
            if len(parts) >= 2:
                metadata["team1"] = parts[0].strip()
                metadata["team2"] = parts[1].strip()
        
        elif section == "season":
            # Extract team, year from season row
            parts = row_text.split("|")
            if len(parts) >= 2:
                metadata["team"] = parts[0].strip()
                metadata["year"] = parts[1].strip()
        
        elif section == "records":
            # Extract category from records row
            parts = row_text.split("|")
            if len(parts) >= 1:
                metadata["category"] = parts[0].strip()
        
        return metadata
    
    def is_code_section(self, text: str) -> bool:
        """Check if text is code/architecture section to skip"""
        code_indicators = ["def ", "class ", "import ", "LangGraph", "workflow", "node(", "graph."]
        return any(indicator in text for indicator in code_indicators)
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF file and return document chunks with row-based chunking"""
        text = self.load_pdf(pdf_path)
        documents = []
        
        # Split into lines for row-based processing
        lines = text.split("\n")
        current_section = "general"
        chunk_id = 0
        
        # Build DataFrame for numeric filtering
        df_data = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip code sections
            if self.is_code_section(line):
                continue
            
            # Detect section type
            detected_section = self.classify_chunk(line)
            if detected_section != "general":
                current_section = detected_section
            
            # Parse row and create chunk
            metadata = self.parse_table_row(line, current_section)
            
            # Handle conflict detection (Section 11)
            if "Section 11" in line or "Conflict" in line:
                metadata["source"] = "secondary"
                metadata["conflict"] = "true"
            
            # Create document chunk
            doc = Document(page_content=line, metadata={**metadata, "chunk_id": chunk_id})
            documents.append(doc)
            
            # Add to DataFrame data
            df_data.append({**metadata, "content": line})
            chunk_id += 1
            
            # Handle H2H bidirectional storage
            if current_section == "h2h" and "team1" in metadata and "team2" in metadata:
                swapped_metadata = metadata.copy()
                swapped_metadata["team1"], swapped_metadata["team2"] = swapped_metadata["team2"], swapped_metadata["team1"]
                doc_swapped = Document(page_content=line, metadata={**swapped_metadata, "chunk_id": chunk_id})
                documents.append(doc_swapped)
                df_data.append({**swapped_metadata, "content": line})
                chunk_id += 1
        
        # Create DataFrame for numeric filtering
        self.dataframe = pd.DataFrame(df_data)
        
        # Save DataFrame to pickle file for later use in retrieval
        os.makedirs("./data", exist_ok=True)
        with open("./data/numeric_dataframe.pkl", "wb") as f:
            pickle.dump(self.dataframe, f)
        
        print(f"Created {len(documents)} document chunks (target: 180-220)")
        return documents
