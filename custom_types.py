import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

class RAGUpsertResult(pydantic.BasemMdel):
    ingested: int

class RAGSearchResult(pydantic.BaseModel):
    context: list[str]
    source: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_cotexts: int