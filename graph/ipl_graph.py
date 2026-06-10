from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from nodes.router_node import router_node
from nodes.retrieval_node import retrieval_node
from nodes.generation_node import generation_node
from retriever.base_retriever import BaseRetriever


class GraphState(TypedDict):
    """Shared state object for the IPL RAG graph"""
    question: str
    query_type: str
    documents: List[Document]
    retrieval_scores: List[float]
    generation: str
    sources: List[str]


class IPLRAGGraph:
    """LangGraph workflow for IPL RAG system"""
    
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with nodes and edges"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("router", router_node)
        workflow.add_node("retrieval", lambda state: retrieval_node(state, self.retriever))
        workflow.add_node("generation", generation_node)
        
        # Add edges
        workflow.set_entry_point("router")
        workflow.add_edge("router", "retrieval")
        workflow.add_edge("retrieval", "generation")
        workflow.add_edge("generation", END)
        
        return workflow.compile()
    
    def run(self, question: str) -> dict:
        """Run the RAG graph with a question"""
        inputs = {
            "question": question,
            "query_type": "",
            "documents": [],
            "retrieval_scores": [],
            "generation": "",
            "sources": []
        }
        result = self.graph.invoke(inputs)
        return result
    
    def get_graph_info(self):
        """Get graph information for visualization"""
        return {
            "nodes": ["router", "retrieval", "generation"],
            "edges": [
                ("router", "retrieval"),
                ("retrieval", "generation"),
                ("generation", "END")
            ]
        }
