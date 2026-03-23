import streamlit as st
import asyncio
import inngest
import os
from pathlib import Path
import time
import requests
import logging

logging.basicConfig(
    filename="app_logs.log",
    filemode="a",
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    force= True
)

logger = logging.getLogger(__name__)

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

    logger.info(f"Send url: {file_path}")
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

async def send_rag_query_event(question: str, top_k: int):
    client = get_Inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k
            },
        )
    )
    return result[0]

def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

def fetch_run(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    answer = requests.get(url)
    answer.raise_for_status()
    data = answer.json()
    return data.get("data", [])

def wait_to_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_run(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
            
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out eaiting for run output (last status {last_status})")
        time.sleep(poll_interval_s)


with st.form("ask_question"):
    question = st.text_input("Enter a question")
    top_k = st.selectbox("choose count of source: ", (1,2,3,4,5))
    button_submit = st.form_submit_button("Ask")
        
    if button_submit and question.strip():
        with st.spinner("Sending event and generation answer..."):
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            output = wait_to_run_output(event_id)
            answer = output.get("answer", "(No relevant responce)")
            sources = output.get("source", [])

        st.subheader("Answer")
        st.write(answer)
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")