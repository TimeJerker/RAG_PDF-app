import streamlit as st
import asyncio
import inngest
import os
from pathlib import Path
import time

st.set_page_config(page_title="RAG Ingest app",layout="centered")

@st.cache_resource
def get_Inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

def file_create_and_save_path(file) -> Path:
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.name

    pdf_data = file.read()

    with open(file_path, "wb") as f:
        f.write(pdf_data)

    return file_path

async def send_path_to_rag_ingest_event(path: Path) -> None:
    client = get_Inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data = {
                "pdf_path": str(path.resolve()),
                "source_id": path.name,
            },
        )
    )



st.title("Upload a PDF")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded:
    with st.spinner("Uploading  file..."):
        path_file = file_create_and_save_path(uploaded)
        asyncio.run(send_path_to_rag_ingest_event(path_file))
        time.sleep(0.3)

    st.success("File uploaded!")

question = st.chat_input("Enter a question")

if question:
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("ai"):
        st.write("Hello!")

# with st.form(key="entering question"):
#     submitted = st.form_submit_button("Ask")
    
#     if question:
#         st.write("Nice")
#         # with st.spinner("Sending event and generation answer..."):
#         #     pass
#         st.subheader("Answer")
#         st.write()