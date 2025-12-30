import streamlit as st
import json
from agent import AgenticSearchAgent
import torch

def initialize_session_state():
    if 'agent' not in st.session_state:
        st.session_state.agent =None
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False


def display_memory(memory_dict):

    with st.expander("Agent Memory & Reasoning", expanded=False):

        st.subheader("Reasoning Process")
        reasoning_logs = memory_dict.get("reasoning_log",[])
        if reasoning_logs:
            for entry in reasoning_logs:
                st.text(f"⏱ {entry.get('timestamp', 'N/A')}")
                st.info(entry.get('reasoning', 'No reasoning logged'))
        else:
            st.text("No reasoning logged yet")

        search_history = memory_dict.get("search_history", [])
        if search_history:
            st.subheader("search_history")
            for search in search_history:
                col1, col2 = st.columns([3,1])
                with col1:
                    st.text(f"Query: {search['query']}")
                with col2:
                    st.text(f"Results: {search['result_count']}")

        KnowledgeBase = memory_dict.get("knowledge_base", [])
        if KnowledgeBase:
            st.subheader("Knowledge Base")
            for kb in KnowledgeBase:
                relevance = kb.get('relevance_score', 0)
                st.markdown(f"**Relevance: {relevance:.2f}**")
                st.markdown(f"*Query: {kb.get('query', 'N/A')}*")
                st.write(kb.get('content', 'No content'))
                st.divider()

def main():

    st.set_page_config(
        page_title = "Agentic Search AI System - Qwen",
        layout = "wide"
    )

    initialize_session_state()


    st.title("Agentic Search AI System (Qwen2.5)")


    with st.sidebar:
        st.header("Configuration")

        st.text("Model: Qwen2.5-7B-Instruct")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.text(f"Device: {device}")

        if st.session_state.model_loaded:
            st.header("Model Loaded")

        else:
            st.warning("Model not loaded")

        st.divider()

        st.header("Agent Settings")
        max_searches = st.slider(
            "Max Searches",
            min_value = 1,
            max_value = 5,
            value = 3,
            help="Maximum number of searches to perform"
        )

    col1, col2 = st.columns([4,1])
    with col1:
        st.write("")
        goal = st.text_input(
            "Enter your Question:",
            value= st.session_state.get("current_goal",''),
            placeholder= "e.g., What are the latest developments in quantum computing?",
            key="goal_input"
        )

    with col2:
        st.write("")
        st.write("")
        st.write("")

        process_button = st.button("Ask" , type="primary", use_container_width=True)

    if process_button and goal :

        if st.session_state.agent is None:
            with st.spinner("Loading Qwen2.5-7B model... This may take 30-60 seconds..."):
                try:
                    st.session_state.agent = AgenticSearchAgent()
                    st.session_state.model_loaded = True
                    st.success("Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.stop()

        with st.spinner("Agent is thinking and searching..."):
            try:
                result = st.session_state.agent.process_goal(goal, max_searches=max_searches)

                st.session_state.results_history.insert(0, result)

                st.success("Goal processed successfully!")

                st.header("Answer")
                st.markdown(result["answer"])

                col1, col2 ,col3 = st.columns(3)
                with col1:
                    search_state = "Yes" if result['search_performed'] else "No"
                    st.metric("Search Performed",search_state)

                with col2:
                    if result['search_performed']:
                        st.metric("Search Count", result.get("search_count", 0))
                    else:
                        st.metric("Searches Count", "N/A")
                with col3:
                    knowledge_items = len(result.get('memory', {}).get('knowledge_base', []))
                    st.metric("Knowledge Items", knowledge_items)

                if result.get('memory'):
                    display_memory(result['memory'])

            except Exception as e:
                st.error(f"Error processing goal: {str(e)}")
                st.exception(e)

    if st.session_state.results_history:
        st.divider()
        st.header("History")

        for idx, result in enumerate(st.session_state.results_history):
            with st.expander(f" {result['goal']}", expanded=(idx == 0)):
                answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
                st.markdown(f"**Answer:** {answer_preview}")

                col1, col2 = st.columns(2)
                with col1:
                    search_text = "Yes" if result.get('search_performed', False) else "No"
                    st.text(f"Search: {search_text}")
                with col2:
                    if result.get('search_performed', False):
                        st.text(f"Searches: {result.get('searches_count', 0)}")

                if st.button("Show Full Details", key=f"details_{idx}"):
                    st.json(result)

        if st.button("Clear History"):
            st.session_state.results_history = []
            st.rerun()






if __name__ == "__main__":
    main()