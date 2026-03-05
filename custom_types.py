import pyndatic

class RAGChunkAndSrc(pyndatic.Basemodel):
    chunks: list[str]
    source_id: str = None

class RAGUpsertResult(pyndatic.Basemodel):
    ingested: int

class RAGSearchResult(pyndatic.Basemodel):
    context: list[str]
    source: list[str]

class RAGQueryResult(pyndatic.Basemodel):
    answer: str
    sources: list[str]
    num_cotexts: int