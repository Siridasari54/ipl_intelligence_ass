from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from nodes.router_node import router_node
from nodes.retrieval_node import retrieval_node
from nodes.generation_node import generation_node
from nodes.validation_node import validation_node
from nodes.reranking_node import reranking_node
from nodes.confidence_node import confidence_node
from nodes.specialist_retrieval_nodes import (
    team_retrieval_node,
    batting_retrieval_node,
    bowling_retrieval_node,
    venue_retrieval_node,
    records_retrieval_node,
    h2h_retrieval_node,
    form_retrieval_node,
    general_retrieval_node
)
from retriever.base_retriever import BaseRetriever


class GraphState(TypedDict):
    """Shared state object for the IPL RAG graph"""
    question: str
    query_type: str
    documents: List[Document]
    retrieval_scores: List[float]
    rerank_scores: List[float]
    confidence_level: str
    validation_status: str
    generation: str
    sources: List[str]


class IPLRAGGraph:
    """LangGraph workflow for IPL RAG system"""
    
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        self.graph = self._build_graph()
    
    def route_query(self, state: GraphState) -> str:
        """
        Route the query to the appropriate specialist retrieval node based on query_type.
        
        Returns the name of the specialist retrieval node to use.
        """
        query_type = state["query_type"]
        
        # Map query types to specialist retrieval nodes
        routing_map = {
            "team": "team_retrieval",
            "batting": "batting_retrieval",
            "bowling": "bowling_retrieval",
            "venue": "venue_retrieval",
            "records": "records_retrieval",
            "h2h": "h2h_retrieval",
            "form": "form_retrieval",
            "general": "general_retrieval"
        }
        
        return routing_map.get(query_type, "general_retrieval")
    
    def route_validation(self, state: GraphState) -> str:
        """
        Route based on validation status.
        
        If validation passed, proceed to confidence assessment.
        If validation failed, end the workflow.
        """
        validation_status = state.get("validation_status", "")
        
        if validation_status == "failed":
            return END
        return "confidence"
    
    def route_confidence(self, state: GraphState) -> str:
        """
        Route based on confidence level.
        
        If confidence is low, end the workflow (answer already generated).
        If confidence is medium or high, proceed to generation.
        """
        confidence_level = state.get("confidence_level", "")
        
        if confidence_level == "low":
            return END
        return "generation"
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with nodes and edges"""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("router", router_node)
        
        # Add specialist retrieval nodes
        workflow.add_node("team_retrieval", lambda state: team_retrieval_node(state, self.retriever))
        workflow.add_node("batting_retrieval", lambda state: batting_retrieval_node(state, self.retriever))
        workflow.add_node("bowling_retrieval", lambda state: bowling_retrieval_node(state, self.retriever))
        workflow.add_node("venue_retrieval", lambda state: venue_retrieval_node(state, self.retriever))
        workflow.add_node("records_retrieval", lambda state: records_retrieval_node(state, self.retriever))
        workflow.add_node("h2h_retrieval", lambda state: h2h_retrieval_node(state, self.retriever))
        workflow.add_node("form_retrieval", lambda state: form_retrieval_node(state, self.retriever))
        workflow.add_node("general_retrieval", lambda state: general_retrieval_node(state, self.retriever))
        
        # Add reranking node
        workflow.add_node("reranking", reranking_node)
        
        # Add validation node
        workflow.add_node("validation", validation_node)
        
        # Add confidence assessment node
        workflow.add_node("confidence", confidence_node)
        
        # Add generation node
        workflow.add_node("generation", generation_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router to specialist retrieval nodes
        workflow.add_conditional_edges(
            "router",
            self.route_query,
            {
                "team_retrieval": "team_retrieval",
                "batting_retrieval": "batting_retrieval",
                "bowling_retrieval": "bowling_retrieval",
                "venue_retrieval": "venue_retrieval",
                "records_retrieval": "records_retrieval",
                "h2h_retrieval": "h2h_retrieval",
                "form_retrieval": "form_retrieval",
                "general_retrieval": "general_retrieval"
            }
        )
        
        # Add edges from all specialist retrieval nodes to reranking
        workflow.add_edge("team_retrieval", "reranking")
        workflow.add_edge("batting_retrieval", "reranking")
        workflow.add_edge("bowling_retrieval", "reranking")
        workflow.add_edge("venue_retrieval", "reranking")
        workflow.add_edge("records_retrieval", "reranking")
        workflow.add_edge("h2h_retrieval", "reranking")
        workflow.add_edge("form_retrieval", "reranking")
        workflow.add_edge("general_retrieval", "reranking")
        
        # Add edge from reranking to validation
        workflow.add_edge("reranking", "validation")
        
        # Add conditional edges from validation to confidence assessment or END
        workflow.add_conditional_edges(
            "validation",
            self.route_validation,
            {
                "confidence": "confidence",
                END: END
            }
        )
        
        # Add conditional edges from confidence to generation or END
        workflow.add_conditional_edges(
            "confidence",
            self.route_confidence,
            {
                "generation": "generation",
                END: END
            }
        )
        
        # Add edge from generation to END
        workflow.add_edge("generation", END)
        
        return workflow.compile()
    
    def run(self, question: str) -> dict:
        """Run the RAG graph with a question"""
        inputs = {
            "question": question,
            "query_type": "",
            "documents": [],
            "retrieval_scores": [],
            "rerank_scores": [],
            "confidence_level": "",
            "validation_status": "",
            "generation": "",
            "sources": []
        }
        result = self.graph.invoke(inputs)
        return result
    
    def get_graph_info(self):
        """Get graph information for visualization"""
        return {
            "nodes": [
                "router",
                "team_retrieval",
                "batting_retrieval",
                "bowling_retrieval",
                "venue_retrieval",
                "records_retrieval",
                "h2h_retrieval",
                "form_retrieval",
                "general_retrieval",
                "reranking",
                "validation",
                "confidence",
                "generation"
            ],
            "edges": [
                ("router", "team_retrieval"),
                ("router", "batting_retrieval"),
                ("router", "bowling_retrieval"),
                ("router", "venue_retrieval"),
                ("router", "records_retrieval"),
                ("router", "h2h_retrieval"),
                ("router", "form_retrieval"),
                ("router", "general_retrieval"),
                ("team_retrieval", "reranking"),
                ("batting_retrieval", "reranking"),
                ("bowling_retrieval", "reranking"),
                ("venue_retrieval", "reranking"),
                ("records_retrieval", "reranking"),
                ("h2h_retrieval", "reranking"),
                ("form_retrieval", "reranking"),
                ("general_retrieval", "reranking"),
                ("reranking", "validation"),
                ("validation", "confidence"),
                ("validation", "END"),
                ("confidence", "generation"),
                ("confidence", "END"),
                ("generation", "END")
            ],
            "workflow": "Router → Conditional Routing → Specialist Retrieval → Reranking → Validation → Confidence Assessment → Generation → END"
        }
