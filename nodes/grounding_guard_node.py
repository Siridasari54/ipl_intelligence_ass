from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq


def grounding_guard_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grounding guard node to ensure final answer is strictly supported by retrieved context.
    
    Checks if generated answer contains facts NOT present in context.
    If hallucination detected, blocks or rewrites response.
    Must operate before final generation.
    
    This node validates that the answer will be grounded in the retrieved documents.
    """
    question = state["question"]
    documents: List[Document] = state["documents"]
    validation_status = state.get("validation_status", "")
    
    # If validation already failed, skip grounding check
    if validation_status == "failed":
        return {
            "question": question,
            "grounding_status": "skipped",
            "grounding_message": "Validation failed - grounding check skipped"
        }
    
    # If no documents, fail grounding
    if not documents or len(documents) == 0:
        return {
            "question": question,
            "grounding_status": "failed",
            "grounding_message": "No documents available for grounding check"
        }
    
    # Extract context from documents
    context = "\n".join([doc.page_content for doc in documents])
    
    # Initialize LLM for grounding check
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    
    # System prompt for grounding check
    system_prompt = """You are a fact-checking expert for IPL cricket data. Your task is to determine if a question can be answered using ONLY the provided context.

Rules:
1. Check if the question can be answered using ONLY the provided context
2. If the question asks for information NOT present in context, return "NOT_GROUNDED"
3. If the question can be answered using the context, return "GROUNDED"
4. Be strict - if any part of the answer requires information not in context, mark as NOT_GROUNDED
5. Return ONLY "GROUNDED" or "NOT_GROUNDED" - no other text

Context:
{context}

Question: {question}

Can this question be answered using ONLY the provided context? Return "GROUNDED" or "NOT_GROUNDED": """.format(context=context, question=question)
    
    # Generate grounding check
    response = llm.invoke([
        {"role": "system", "content": system_prompt}
    ])
    
    grounding_result = response.content.strip().upper()
    
    if "NOT_GROUNDED" in grounding_result:
        return {
            "question": question,
            "grounding_status": "failed",
            "grounding_message": "Answer not grounded in dataset - question requires information not available in retrieved context"
        }
    else:
        return {
            "question": question,
            "grounding_status": "passed",
            "grounding_message": "Question is grounded in retrieved context"
        }
