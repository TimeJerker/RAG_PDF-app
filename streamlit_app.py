import streamlit as st
import asyncio
import inngest
import os
from pathlib import Path

st.set_page_config(page_title="RAG Ingest app",layout="centered")

@st.cache_resource
def get_Inngest() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

def save_uploaded_pdf_file(file) -> Path:
    path_dir = Path(uploaded)

st.title("Upload a PDF")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

with st.form(key="entering question"):
    question = st.text_input("Enter a question")
    submitted = st.form_submit_button("Ask")
    
    if question.strip():
        with st.spinner("Sending event and generation answer..."):
            pass
        st.subheader("Answer")
        st.write()