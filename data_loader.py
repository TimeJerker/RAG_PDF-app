from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv
import os
import json
from pathlib import Path

load_dotenv()

token = os.getenv("API_TOKEN")

login(
    token= token,
    add_to_git_credential = True 
)

EMBED_MODEL = "intfloat/multilingual-e5-large"

#размер вектора нейронки
EMBED_DIM = 1024


model = SentenceTransformer(EMBED_MODEL)

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    path_dir = Path.cwd() / "chunks"
    number = len(list(path_dir.glob("chunk_*.json"))) + 1
    try:  
        os.makedirs("chunks")
        chunks_path = str(os.path.join(os.getcwd(),"chunks", f"chunk_{number}.json"))
    except FileExistsError:
        chunks_path = str(os.path.join(os.getcwd(),"chunks", f"chunk_{number}.json"))
        
    with open(chunks_path, "w", encoding="utf-8") as ch:
        json.dump(chunks, ch, ensure_ascii=False)
    return chunks_path, len(chunks)

def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    #некоторые модели требуют префикс для лучших результатов
    if is_query:
        # Это поисковый запрос
        prefixed_texts = [f"query: {text}" for text in texts]
    else:
        # Это документы для индексации
        prefixed_texts = [f"passage: {text}" for text in texts]
    
    embedding = model.encode(
        prefixed_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

    return embedding.tolist()
