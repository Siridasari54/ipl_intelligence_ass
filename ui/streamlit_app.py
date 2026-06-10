import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from vectorstore.chroma_store import ChromaVectorStore
from retriever.base_retriever import BaseRetriever
from graph.ipl_graph import IPLRAGGraph

load_dotenv()


def main():
    """Main Streamlit UI for IPL Intelligence Assistant"""
    st.set_page_config(
        page_title="IPL Intelligence Assistant",
        page_icon="🏏",
        layout="wide"
    )
    
    st.title("🏏 IPL Intelligence Assistant")
    st.markdown("---")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Info")
        st.info("LangGraph-based RAG System")
        
        st.subheader("Query Types")
        st.markdown("""
        - **Team**: Team-related queries
        - **Batting**: Batting statistics
        - **Bowling**: Bowling statistics
        - **Venue**: Stadium/venue info
        - **Records**: Records & milestones
        - **General**: General IPL queries
        """)
        
        st.subheader("Graph Workflow")
        st.markdown("""
        1. **Router** - Classify query type
        2. **Retrieval** - Fetch relevant chunks
        3. **Generation** - Generate answer
        """)
    
    # Initialize components
    @st.cache_resource
    def initialize_system():
        """Initialize vector store and graph"""
        vectorstore = ChromaVectorStore(persist_directory="./data/chroma_db")
        
        # Check if vector store exists
        if os.path.exists("./data/chroma_db"):
            vectorstore.load_vectorstore()
        else:
            st.error("Vector store not found. Please run the ingestion script first.")
            st.stop()
        
        retriever = BaseRetriever(vectorstore)
        graph = IPLRAGGraph(retriever)
        return graph
    
    try:
        graph = initialize_system()
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.info("Please ensure you have run the ingestion script first.")
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "query_type" in message:
                st.caption(f"📌 Query Type: {message['query_type'].upper()}")
            
            if "retrieved_chunks" in message and message["retrieved_chunks"]:
                with st.expander("📄 Retrieved Context"):
                    for i, chunk in enumerate(message["retrieved_chunks"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(chunk)
                        st.caption(f"Score: {message['scores'][i-1]:.4f}")
            
            if "sources" in message and message["sources"]:
                st.caption(f"📚 Sources: {', '.join(message['sources'])}")
    
    # Chat input
    if prompt := st.chat_input("Ask about IPL cricket..."):
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Run the graph
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    result = graph.run(prompt)
                    
                    # Display query type
                    query_type = result.get("query_type", "general")
                    st.caption(f"📌 Query Type: {query_type.upper()}")
                    
                    # Display retrieved context
                    documents = result.get("documents", [])
                    scores = result.get("retrieval_scores", [])
                    
                    if documents:
                        with st.expander("📄 Retrieved Context"):
                            for i, doc in enumerate(documents, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.text(doc.page_content)
                                if i <= len(scores):
                                    st.caption(f"Score: {scores[i-1]:.4f}")
                    
                    # Display answer
                    answer = result.get("generation", "No answer generated.")
                    st.markdown(answer)
                    
                    # Display sources
                    sources = result.get("sources", [])
                    if sources:
                        st.caption(f"📚 Sources: {', '.join(set(sources))}")
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "query_type": query_type,
                        "retrieved_chunks": [doc.page_content for doc in documents],
                        "scores": scores,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
