import os
from typing import Any

import chromadb
from dotenv import load_dotenv

load_dotenv()


class ChromaDBClient:
    def __init__(self):
        self.chroma_dir = os.getenv("CHROMA_DIR", "./local/chroma")
        self.collection_name = os.getenv("CHROMA_COLLECTION", "wiki_chunks")

        self.hnsw_construction_ef = int(os.getenv("CHROMA_HNSW_CONSTRUCTION_EF", "200"))
        self.hnsw_search_ef = int(os.getenv("CHROMA_HNSW_SEARCH_EF", "100"))

        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": self.hnsw_construction_ef,
                "hnsw:search_ef": self.hnsw_search_ef,
            },
        )

    def retrieve(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        top_k: int = 25,
    ) -> list[dict[str, Any]]:
        if query_text is None and query_embedding is None:
            raise ValueError("Either query_text or query_embedding must be provided")

        if query_embedding is not None:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        return [
            {
                "id": chunk_id,
                "text": text,
                "metadata": metadata or {},
                "score": 1 / (1 + distance),
                "source": "chroma",
            }
            for chunk_id, text, metadata, distance in zip(
                ids, docs, metadatas, distances
            )
        ]


chromadb_client = ChromaDBClient()