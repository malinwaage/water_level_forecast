import streamlit as st

def setup_page_config():
    st.set_page_config(
        page_title="Sogndalsvatn Prediction System",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
