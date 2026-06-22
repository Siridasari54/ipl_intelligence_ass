"""
Test script to verify the enhanced LangGraph workflow with advanced RAG features
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vectorstore.chroma_store import ChromaVectorStore
from retriever.base_retriever import BaseRetriever
from graph.ipl_graph import IPLRAGGraph

load_dotenv()


def test_graph_build():
    """Test that the graph builds correctly with new nodes"""
    print("Testing graph build...")
    try:
        vectorstore = ChromaVectorStore(persist_directory="./data/chroma_db")
        
        if os.path.exists("./data/chroma_db"):
            vectorstore.load_vectorstore()
        else:
            print("Vector store not found. Skipping test.")
            return None
        
        retriever = BaseRetriever(vectorstore)
        graph = IPLRAGGraph(retriever)
        
        # Get graph info
        graph_info = graph.get_graph_info()
        print(f"✓ Graph built successfully")
        print(f"  Nodes: {len(graph_info['nodes'])}")
        print(f"  Expected nodes: 13 (router, 8 specialists, reranking, validation, confidence, generation)")
        print(f"  Edges: {len(graph_info['edges'])}")
        print(f"  Workflow: {graph_info['workflow']}")
        
        # Verify new nodes are present
        expected_nodes = ["reranking", "confidence"]
        for node in expected_nodes:
            if node in graph_info['nodes']:
                print(f"  ✓ Node '{node}' present")
            else:
                print(f"  ✗ Node '{node}' missing")
        
        return graph
    except Exception as e:
        print(f"✗ Graph build failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_query_routing(graph):
    """Test that query routing works for different query types"""
    print("\nTesting query routing...")
    
    test_queries = [
        ("Tell me about Mumbai Indians", "team"),
        ("What is Virat Kohli's batting average?", "batting"),
        ("Who has the most wickets in IPL?", "bowling"),
        ("Tell me about Eden Gardens stadium", "venue"),
        ("What is the highest individual score in IPL?", "records"),
        ("How have CSK and MI performed against each other?", "h2h"),
        ("What is RCB's recent form?", "form"),
        ("When did IPL start?", "general")
    ]
    
    for query, expected_type in test_queries:
        try:
            result = graph.run(query)
            query_type = result.get("query_type", "")
            
            if query_type == expected_type:
                print(f"✓ Query routed correctly: '{query[:40]}...' -> {query_type}")
            else:
                print(f"⚠ Query routed to {query_type}, expected {expected_type}: '{query[:40]}...'")
        except Exception as e:
            print(f"✗ Query failed: '{query[:40]}...' - {str(e)}")


def test_reranking(graph):
    """Test that reranking node works correctly"""
    print("\nTesting reranking node...")
    
    try:
        result = graph.run("Tell me about Mumbai Indians")
        rerank_scores = result.get("rerank_scores", [])
        
        if rerank_scores:
            print(f"✓ Reranking node executed successfully")
            print(f"  Number of rerank scores: {len(rerank_scores)}")
            print(f"  Rerank scores: {[f'{s:.4f}' for s in rerank_scores]}")
        else:
            print(f"⚠ No rerank scores returned")
    except Exception as e:
        print(f"✗ Reranking test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_confidence_assessment(graph):
    """Test that confidence assessment node works correctly"""
    print("\nTesting confidence assessment node...")
    
    try:
        result = graph.run("Tell me about Mumbai Indians")
        confidence_level = result.get("confidence_level", "")
        
        if confidence_level in ["high", "medium", "low"]:
            print(f"✓ Confidence assessment node executed: level = {confidence_level}")
        else:
            print(f"⚠ Confidence level unexpected: {confidence_level}")
    except Exception as e:
        print(f"✗ Confidence assessment test failed: {str(e)}")


def test_metadata_filtering(graph):
    """Test that metadata filtering works correctly"""
    print("\nTesting metadata filtering...")
    
    try:
        result = graph.run("Tell me about Mumbai Indians")
        documents = result.get("documents", [])
        
        if documents:
            metadata_sections = [doc.metadata.get("metadata_section", "unknown") for doc in documents]
            print(f"✓ Metadata filtering executed")
            print(f"  Metadata sections: {metadata_sections}")
        else:
            print(f"⚠ No documents returned")
    except Exception as e:
        print(f"✗ Metadata filtering test failed: {str(e)}")


def test_hybrid_retrieval(graph):
    """Test that hybrid retrieval (Chroma + BM25) works correctly"""
    print("\nTesting hybrid retrieval...")
    
    try:
        result = graph.run("Tell me about Mumbai Indians")
        retrieval_scores = result.get("retrieval_scores", [])
        
        if retrieval_scores:
            print(f"✓ Hybrid retrieval executed successfully")
            print(f"  Number of retrieval scores: {len(retrieval_scores)}")
            print(f"  Retrieval scores: {[f'{s:.4f}' for s in retrieval_scores]}")
        else:
            print(f"⚠ No retrieval scores returned")
    except Exception as e:
        print(f"✗ Hybrid retrieval test failed: {str(e)}")


def test_validation_node(graph):
    """Test that validation node works correctly"""
    print("\nTesting validation node...")
    
    try:
        result = graph.run("Tell me about Mumbai Indians")
        validation_status = result.get("validation_status", "")
        
        if validation_status in ["passed", "failed"]:
            print(f"✓ Validation node executed: status = {validation_status}")
        else:
            print(f"⚠ Validation status unexpected: {validation_status}")
    except Exception as e:
        print(f"✗ Validation test failed: {str(e)}")


def test_generation_node(graph):
    """Test that generation node handles confidence level"""
    print("\nTesting generation node...")
    
    try:
        result = graph.run("Tell me about Mumbai Indians")
        generation = result.get("generation", "")
        
        if generation:
            print(f"✓ Generation node executed successfully")
            print(f"  Generated response length: {len(generation)} characters")
        else:
            print(f"⚠ Generation node returned empty response")
    except Exception as e:
        print(f"✗ Generation test failed: {str(e)}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Enhanced IPL LangGraph RAG System")
    print("With Advanced RAG Features")
    print("=" * 60)
    
    # Test graph build
    graph = test_graph_build()
    
    if graph is None:
        print("\nCannot proceed with tests without graph.")
        return
    
    # Test query routing
    test_query_routing(graph)
    
    # Test new features
    test_hybrid_retrieval(graph)
    test_metadata_filtering(graph)
    test_reranking(graph)
    test_confidence_assessment(graph)
    
    # Test existing features
    test_validation_node(graph)
    test_generation_node(graph)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("All components tested. Check results above.")
    print("\nTo run the full application:")
    print("  streamlit run ui/streamlit_app.py")
    print("\nTo re-ingest data with metadata tags:")
    print("  python ingest.py")


if __name__ == "__main__":
    main()
