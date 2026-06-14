# src/reranker.py

import os
from typing import Any

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest

load_dotenv()


class Reranker:
    def __init__(self):
        self.model_name = os.getenv(
            "RERANKER_MODEL",
            "ms-marco-MiniLM-L-12-v2",
        )

        self.ranker = Ranker(model_name=self.model_name)

    def rerank(
        self,
        query: str,
        passages: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not passages:
            return []

        request = RerankRequest(
            query=query,
            passages=[
                {
                    "id": passage.get("id"),
                    "text": passage.get("text", ""),
                    "metadata": passage.get("metadata", {}),
                    "original_score": passage.get("score", 0.0),
                    "retrieval_sources": passage.get("retrieval_sources", []),
                    "semantic_score": passage.get("semantic_score", 0.0),
                    "keyword_score": passage.get("keyword_score", 0.0),
                }
                for passage in passages
            ],
        )

        ranked = self.ranker.rerank(request)

        results = []

        for item in ranked[:top_k]:
            results.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text", ""),
                    "metadata": item.get("metadata", {}),
                    "score": item.get("score", 0.0),
                    "rerank_score": item.get("score", 0.0),
                    "original_score": item.get("original_score", 0.0),
                    "retrieval_sources": item.get("retrieval_sources", []),
                    "semantic_score": item.get("semantic_score", 0.0),
                    "keyword_score": item.get("keyword_score", 0.0),
                    "source": "reranker",
                }
            )

        return results


reranker = Reranker()