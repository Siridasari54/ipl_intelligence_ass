import os
from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generation node to create answer from retrieved context.
    
    Generates answers strictly from retrieved context with source citations.
    """
    question = state["question"]
    query_type = state["query_type"]
    documents: List[Document] = state["documents"]
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Create generation prompt
    prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant specializing in IPL (Indian Premier League) cricket information.
        Your role is to answer questions based ONLY on the provided context.
        
        Query Type: {query_type}
        
        Context:
        {context}
        
        Question:
        {question}
        
        Instructions:
        1. Answer the question using ONLY the information from the context above.
        2. If the answer is not in the context, say "I don't have enough information from the dataset to answer this question."
        3. Be concise and informative.
        4. Include source citations in your answer using [Source X] format where X is the chunk number.
        5. Do not make up or hallucinate information.
        
        Answer:"""
    )
    
    # Format documents with source numbers
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Source {i}] {doc.page_content}\n(Source: {source})")
    
    context = "\n\n".join(context_parts)
    
    # Generate response
    chain = prompt | llm
    response = chain.invoke({
        "query_type": query_type,
        "context": context,
        "question": question
    })
    
    # Extract sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in documents]))
    
    return {
        "question": question,
        "query_type": query_type,
        "documents": documents,
        "generation": response.content,
        "sources": sources
    }
