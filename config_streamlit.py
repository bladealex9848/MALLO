import streamlit as st

def set_streamlit_page_config():
    st.set_page_config(
        page_title="MALLO: MultiAgent LLM Orchestrator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# MALLO: MultiAgent LLM Orchestrator\n\nThis is an advanced system for orchestrating multiple Large Language Model agents."
        }
    )