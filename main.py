import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
from qdrant_bd import QdrantStorage
from data_loader import load_and_chunk_pdf, embed_texts
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGPathChunks
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import json

load_dotenv()

ollama = init_chat_model(
    model="qwen3.5:9b",
    model_provider="ollama",
    temperature=0.1,
    max_tokens=512,
    base_url="http://127.0.0.1:11434"
)

inngest_client = inngest.Inngest(
    app_id="rag-app",
    logger = logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: inngest PDF",
    trigger= inngest.TriggerEvent(event="rag/ingest_pdf")
    )
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGPathChunks:
        pdf_path = ctx.event.data["pdf_path"] #достаем из ввода пользоваетля например: "pdf_path": "/documents/report.pdf"
        source_id = ctx.event.data.get("source_id", pdf_path)
        path_chunks, len_chunks = load_and_chunk_pdf(pdf_path)
        return RAGPathChunks(path=path_chunks,len_chunks=len_chunks,source_id=source_id)
        #return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
    
    def _upsert(chunks_and_src: RAGPathChunks) -> RAGUpsertResult:
        chunks_path = chunks_and_src.path
        source_id = chunks_and_src.source_id
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payload = [{"source":source_id,"text":chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids,vecs,payload)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGPathChunks)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger = inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question], True)[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question,top_k), output_type=RAGSearchResult)
    
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты отвечаешь на вопросы только на основе предоставленного контекста на русском."),
        ("human", "Используй следующий контекст для ответа на вопрос.\n\n"
        "Контекст:\n{context_block}\n\n"
        "Вопрос: {question}\n"
        "Ответь кратко, используя только контекст выше.")
    ])
    
    chain = prompt | ollama

    response = await chain.ainvoke({ #асинхронный вызов ainvoke
        "context_block":context_block,
        "question": question
    })

    answer = response.content.strip()
    return {"answer": answer, 
            "sources": found.sources, 
            "num_contexts": len(found.contexts)
    }

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])


