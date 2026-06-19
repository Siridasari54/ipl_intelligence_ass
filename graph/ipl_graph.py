from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from nodes.router_node import router_node
from nodes.query_rewrite_node import query_rewrite_node
from nodes.query_decomposition_node import query_decomposition_node
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
    rewritten_query: str
    sub_queries: List[str]
    query_type: str
    documents: List[Document]
    retrieval_scores: List[float]
    rerank_scores: List[float]
    confidence_level: str
    validation_status: str
    generation: str
    sources: List[str]
    use_parallel: bool


class IPLRAGGraph:
    """LangGraph workflow for IPL RAG system"""
    
    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever
        self.graph = self._build_graph()
    
    def route_query(self, state: GraphState) -> str:
        """
        Route the query to the appropriate specialist retrieval node based on query_type.
        Also determines if parallel execution is needed for complex queries.
        
        Returns the name of the specialist retrieval node to use.
        """
        query_type = state["query_type"]
        question = state["question"].lower()
        
        # Detect if parallel execution is needed (Dream11/prediction queries)
        parallel_keywords = ["dream11", "prediction", "predict", "best xi", "fantasy", "team combination", "compare"]
        use_parallel = any(keyword in question for keyword in parallel_keywords)
        
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
        
        if use_parallel:
            return "parallel_retrieval"
        
        return routing_map.get(query_type, "general_retrieval")
    
    def parallel_retrieval_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Execute multiple specialist retrieval nodes in parallel for complex queries.
        Merges all outputs into a single context.
        """
        sub_queries = state.get("sub_queries", [state["rewritten_query"]])
        all_documents = []
        all_scores = []
        
        # Map sub-queries to appropriate specialist nodes
        specialist_nodes = {
            "team": team_retrieval_node,
            "batting": batting_retrieval_node,
            "bowling": bowling_retrieval_node,
            "venue": venue_retrieval_node,
            "records": records_retrieval_node,
            "h2h": h2h_retrieval_node,
            "form": form_retrieval_node,
            "general": general_retrieval_node
        }
        
        # Execute retrieval for each sub-query
        for sub_query in sub_queries:
            # Determine query type for this sub-query
            temp_state = state.copy()
            temp_state["question"] = sub_query
            
            # Use router to determine query type
            from nodes.router_node import router_node
            routed = router_node(temp_state)
            query_type = routed["query_type"]
            
            # Execute appropriate specialist node
            if query_type in specialist_nodes:
                result = specialist_nodes[query_type](temp_state, self.retriever)
                all_documents.extend(result["documents"])
                all_scores.extend(result["retrieval_scores"])
        
        return {
            "question": state["question"],
            "rewritten_query": state["rewritten_query"],
            "sub_queries": sub_queries,
            "query_type": state["query_type"],
            "documents": all_documents,
            "retrieval_scores": all_scores,
            "use_parallel": True
        }
    
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
        workflow.add_node("query_rewrite", query_rewrite_node)
        workflow.add_node("query_decomposition", query_decomposition_node)
        
        # Add specialist retrieval nodes
        workflow.add_node("team_retrieval", lambda state: team_retrieval_node(state, self.retriever))
        workflow.add_node("batting_retrieval", lambda state: batting_retrieval_node(state, self.retriever))
        workflow.add_node("bowling_retrieval", lambda state: bowling_retrieval_node(state, self.retriever))
        workflow.add_node("venue_retrieval", lambda state: venue_retrieval_node(state, self.retriever))
        workflow.add_node("records_retrieval", lambda state: records_retrieval_node(state, self.retriever))
        workflow.add_node("h2h_retrieval", lambda state: h2h_retrieval_node(state, self.retriever))
        workflow.add_node("form_retrieval", lambda state: form_retrieval_node(state, self.retriever))
        workflow.add_node("general_retrieval", lambda state: general_retrieval_node(state, self.retriever))
        
        # Add parallel retrieval node
        workflow.add_node("parallel_retrieval", self.parallel_retrieval_node)
        
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
        
        # Add edge from router to query rewrite
        workflow.add_edge("router", "query_rewrite")
        
        # Add edge from query rewrite to query decomposition
        workflow.add_edge("query_rewrite", "query_decomposition")
        
        # Add conditional edges from query decomposition to retrieval nodes
        workflow.add_conditional_edges(
            "query_decomposition",
            self.route_query,
            {
                "team_retrieval": "team_retrieval",
                "batting_retrieval": "batting_retrieval",
                "bowling_retrieval": "bowling_retrieval",
                "venue_retrieval": "venue_retrieval",
                "records_retrieval": "records_retrieval",
                "h2h_retrieval": "h2h_retrieval",
                "form_retrieval": "form_retrieval",
                "general_retrieval": "general_retrieval",
                "parallel_retrieval": "parallel_retrieval"
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
        workflow.add_edge("parallel_retrieval", "reranking")
        
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
            "rewritten_query": "",
            "sub_queries": [],
            "query_type": "",
            "documents": [],
            "retrieval_scores": [],
            "rerank_scores": [],
            "confidence_level": "",
            "validation_status": "",
            "generation": "",
            "sources": [],
            "use_parallel": False
        }
        result = self.graph.invoke(inputs)
        return result
    
    def get_graph_info(self):
        """Get graph information for visualization"""
        return {
            "nodes": [
                "router",
                "query_rewrite",
                "query_decomposition",
                "team_retrieval",
                "batting_retrieval",
                "bowling_retrieval",
                "venue_retrieval",
                "records_retrieval",
                "h2h_retrieval",
                "form_retrieval",
                "general_retrieval",
                "parallel_retrieval",
                "reranking",
                "validation",
                "confidence",
                "generation"
            ],
            "edges": [
                ("router", "query_rewrite"),
                ("query_rewrite", "query_decomposition"),
                ("query_decomposition", "team_retrieval"),
                ("query_decomposition", "batting_retrieval"),
                ("query_decomposition", "bowling_retrieval"),
                ("query_decomposition", "venue_retrieval"),
                ("query_decomposition", "records_retrieval"),
                ("query_decomposition", "h2h_retrieval"),
                ("query_decomposition", "form_retrieval"),
                ("query_decomposition", "general_retrieval"),
                ("query_decomposition", "parallel_retrieval"),
                ("team_retrieval", "reranking"),
                ("batting_retrieval", "reranking"),
                ("bowling_retrieval", "reranking"),
                ("venue_retrieval", "reranking"),
                ("records_retrieval", "reranking"),
                ("h2h_retrieval", "reranking"),
                ("form_retrieval", "reranking"),
                ("general_retrieval", "reranking"),
                ("parallel_retrieval", "reranking"),
                ("reranking", "validation"),
                ("validation", "confidence"),
                ("validation", "END"),
                ("confidence", "generation"),
                ("confidence", "END"),
                ("generation", "END")
            ],
            "workflow": "Router → QueryRewrite → QueryDecomposition → Parallel Specialist Retrieval Nodes → Rerank → Validation → Confidence → Generation → END"
        }
