import os
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


class QueryProcessor:
    def __init__(self):
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        )

        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def normalize_query(self, query: str) -> str:
        return query.strip()

    def rewrite_query(self, query: str) -> str:
        """
        Placeholder for future query rewriting.

        Later this can use:
        - LangChain
        - LLM-based rewriting
        - HyDE
        - multi-query expansion
        """
        return query

    def embed_query(self, query: str) -> list[float]:
        normalized_query = self.normalize_query(query)
        rewritten_query = self.rewrite_query(normalized_query)

        embedding = self.embedding_model.encode(
            rewritten_query,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def process(self, query: str) -> dict:
        normalized_query = self.normalize_query(query)
        rewritten_query = self.rewrite_query(normalized_query)
        embedding = self.embed_query(rewritten_query)

        return {
            "original_query": query,
            "query": rewritten_query,
            "embedding": embedding,
        }


query_processor = QueryProcessor()