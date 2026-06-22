import re
import pandas as pd
from typing import List, Dict, Any, Optional


def detect_numeric_query(query: str) -> bool:
    """Detect if query contains numeric conditions using regex/keyword matching"""
    numeric_patterns = [
        r'economy\s*<\s*\d+',  # economy < X
        r'economy\s*less\s+than\s*\d+',
        r'strike\s*rate\s*>\s*\d+',  # strike_rate > X
        r'strike\s*rate\s*greater\s+than\s*\d+',
        r'average\s*>\s*\d+',  # average > X
        r'average\s*greater\s+than\s*\d+',
        r'role\s*==?\s*["\']?Opener["\']?',  # role == "Opener"
        r'role\s*is\s+Opener',
        r'titles\s*>\s*\d+',  # titles > X
        r'titles\s*greater\s+than\s*\d+',
        r'avg_first_innings\s*>\s*\d+',  # avg_first_innings > X
        r'avg_first_innings\s*greater\s+than\s*\d+',
        r'\d+\s+wickets?',  # numeric wickets
        r'\d+\s+runs?',  # numeric runs
    ]
    query_lower = query.lower()
    return any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in numeric_patterns)


def parse_numeric_filters(query: str, query_type: str) -> Dict[str, Any]:
    """Parse numeric conditions from query string"""
    filters = {}
    
    # economy < X
    economy_match = re.search(r'economy\s*<\s*(\d+\.?\d*)', query, re.IGNORECASE)
    if economy_match:
        filters['economy_max'] = float(economy_match.group(1))
    
    # strike_rate > X
    sr_match = re.search(r'strike\s*rate\s*>\s*(\d+\.?\d*)', query, re.IGNORECASE)
    if sr_match:
        filters['strike_rate_min'] = float(sr_match.group(1))
    
    # average > X
    avg_match = re.search(r'average\s*>\s*(\d+\.?\d*)', query, re.IGNORECASE)
    if avg_match:
        filters['average_min'] = float(avg_match.group(1))
    
    # role == "Opener"
    role_match = re.search(r'role\s*==?\s*["\']?(Opener|Middle-order|WK-Bat|All-rounder|WK-Finisher)["\']?', query, re.IGNORECASE)
    if role_match:
        filters['role'] = role_match.group(1)
    
    # titles > X
    titles_match = re.search(r'titles\s*>\s*(\d+)', query, re.IGNORECASE)
    if titles_match:
        filters['titles_min'] = int(titles_match.group(1))
    
    # avg_first_innings > X
    avg_first_match = re.search(r'avg_first_innings\s*>\s*(\d+\.?\d*)', query, re.IGNORECASE)
    if avg_first_match:
        filters['avg_first_innings_min'] = float(avg_first_match.group(1))
    
    return filters


def numeric_filter(dataframe: pd.DataFrame, query_type: str, filters: Dict[str, Any]) -> List[str]:
    """Apply numeric filters to DataFrame and return matching rows as text"""
    if dataframe is None or dataframe.empty:
        return []
    
    filtered_df = dataframe.copy()
    
    # Apply filters based on query type
    if query_type == "bowling":
        if 'economy_max' in filters:
            # Extract economy from content if numeric column exists
            if 'economy' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['economy'] < filters['economy_max']]
    
    if query_type in ["batting", "bowling"]:
        if 'average_min' in filters:
            if 'average' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['average'] > filters['average_min']]
    
    if query_type == "batting":
        if 'strike_rate_min' in filters:
            if 'strike_rate' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['strike_rate'] > filters['strike_rate_min']]
        if 'role' in filters:
            if 'role' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['role'] == filters['role']]
    
    if query_type == "team":
        if 'titles_min' in filters:
            if 'titles' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['titles'] > filters['titles_min']]
    
    if query_type == "venue":
        if 'avg_first_innings_min' in filters:
            if 'avg_first_innings' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['avg_first_innings'] > filters['avg_first_innings_min']]
    
    # Return matching rows as text
    return filtered_df['content'].tolist() if 'content' in filtered_df.columns else []
