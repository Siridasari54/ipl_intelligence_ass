import re
from typing import Dict, Any, List
from langchain_core.documents import Document


def extract_numeric_values(text: str) -> Dict[str, float]:
    """Extract numeric values for common IPL fields from text"""
    values = {}
    
    # Extract runs
    runs_match = re.search(r'runs[:\s]+(\d+)', text, re.IGNORECASE)
    if runs_match:
        values['runs'] = float(runs_match.group(1))
    
    # Extract wickets
    wickets_match = re.search(r'wickets?[:\s]+(\d+)', text, re.IGNORECASE)
    if wickets_match:
        values['wickets'] = float(wickets_match.group(1))
    
    # Extract matches
    matches_match = re.search(r'matches?[:\s]+(\d+)', text, re.IGNORECASE)
    if matches_match:
        values['matches'] = float(matches_match.group(1))
    
    # Extract average
    avg_match = re.search(r'average[:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
    if avg_match:
        values['average'] = float(avg_match.group(1))
    
    # Extract strike rate
    sr_match = re.search(r'strike\s*rate[:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
    if sr_match:
        values['strike_rate'] = float(sr_match.group(1))
    
    # Extract economy
    econ_match = re.search(r'economy[:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
    if econ_match:
        values['economy'] = float(econ_match.group(1))
    
    return values


def detect_conflicts(documents: List[Document]) -> Dict[str, Any]:
    """
    Detect conflicting values across primary and secondary sources.
    
    Compares retrieved chunks across primary and secondary sources
    for same entity + same field (runs, wickets, matches, etc.)
    """
    primary_docs = [doc for doc in documents if doc.metadata.get("source") == "primary"]
    secondary_docs = [doc for doc in documents if doc.metadata.get("source") == "secondary"]
    
    conflicts = []
    conflict_detected = False
    
    # Extract entities from documents (simplified - using player_name if available)
    primary_entities = {}
    secondary_entities = {}
    
    for doc in primary_docs:
        entity = doc.metadata.get("player_name") or doc.metadata.get("team") or doc.metadata.get("venue_name")
        if entity:
            values = extract_numeric_values(doc.page_content)
            if entity not in primary_entities:
                primary_entities[entity] = {}
            primary_entities[entity].update(values)
    
    for doc in secondary_docs:
        entity = doc.metadata.get("player_name") or doc.metadata.get("team") or doc.metadata.get("venue_name")
        if entity:
            values = extract_numeric_values(doc.page_content)
            if entity not in secondary_entities:
                secondary_entities[entity] = {}
            secondary_entities[entity].update(values)
    
    # Compare values for same entity
    for entity in primary_entities:
        if entity in secondary_entities:
            for field in primary_entities[entity]:
                if field in secondary_entities[entity]:
                    primary_val = primary_entities[entity][field]
                    secondary_val = secondary_entities[entity][field]
                    
                    # Check for significant difference (more than 5% or 1 unit)
                    if abs(primary_val - secondary_val) > max(0.05 * primary_val, 1.0):
                        conflict_detected = True
                        conflicts.append({
                            "entity": entity,
                            "field": field,
                            "primary_value": primary_val,
                            "secondary_value": secondary_val,
                            "difference": abs(primary_val - secondary_val)
                        })
    
    return {
        "conflict_detected": conflict_detected,
        "conflicts": conflicts
    }


def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validation node to check if sufficient context was retrieved and detect conflicts.
    
    Validates:
    - Whether documents were retrieved
    - Whether sufficient context exists
    - Whether answer generation should continue
    - Whether conflicting data exists between primary and secondary sources
    
    If validation fails, returns a message indicating insufficient information.
    If conflicts detected, stores conflict information for downstream nodes.
    """
    question = state["question"]
    query_type = state["query_type"]
    documents: List[Document] = state["documents"]
    retrieval_scores: List[float] = state["retrieval_scores"]
    
    # Check if documents were retrieved
    if not documents or len(documents) == 0:
        validation_status = "failed"
        generation = "I don't have enough information from the indexed IPL dataset to answer this question."
        return {
            "question": question,
            "query_type": query_type,
            "documents": documents,
            "retrieval_scores": retrieval_scores,
            "validation_status": validation_status,
            "generation": generation,
            "sources": [],
            "conflict_detected": False,
            "conflicts": []
        }
    
    # Check if retrieval scores indicate low relevance (threshold: 0.3)
    avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
    if avg_score < 0.3:
        validation_status = "failed"
        generation = "I don't have enough information from the indexed IPL dataset to answer this question."
        return {
            "question": question,
            "query_type": query_type,
            "documents": documents,
            "retrieval_scores": retrieval_scores,
            "validation_status": validation_status,
            "generation": generation,
            "sources": [],
            "conflict_detected": False,
            "conflicts": []
        }
    
    # Check if document content is sufficient (at least 50 characters total)
    total_content_length = sum(len(doc.page_content) for doc in documents)
    if total_content_length < 50:
        validation_status = "failed"
        generation = "I don't have enough information from the indexed IPL dataset to answer this question."
        return {
            "question": question,
            "query_type": query_type,
            "documents": documents,
            "retrieval_scores": retrieval_scores,
            "validation_status": validation_status,
            "generation": generation,
            "sources": [],
            "conflict_detected": False,
            "conflicts": []
        }
    
    # Detect conflicts between primary and secondary sources
    conflict_info = detect_conflicts(documents)
    
    # Validation passed
    validation_status = "passed"
    result = {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "retrieval_scores": retrieval_scores,
        "validation_status": validation_status,
        "conflict_detected": conflict_info["conflict_detected"],
        "conflicts": conflict_info["conflicts"]
    }
    
    # Add conflict explanation if conflicts detected
    if conflict_info["conflict_detected"]:
        conflict_explanation = f"Warning: Conflicting data detected between primary and secondary sources for {len(conflict_info['conflicts'])} field(s). "
        for conflict in conflict_info["conflicts"]:
            conflict_explanation += f"{conflict['entity']} {conflict['field']}: primary={conflict['primary_value']}, secondary={conflict['secondary_value']}. "
        result["conflict_explanation"] = conflict_explanation
    
    return result
