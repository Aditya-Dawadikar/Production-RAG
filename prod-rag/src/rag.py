# src/rag.py

import os
from typing import Any

from dotenv import load_dotenv

from src.llm_client import llm_client
from src.reranker import reranker
from src.retriever import retriever

load_dotenv()


class RAGClient:
    def __init__(self):
        self.retriever = retriever
        self.reranker = reranker
        self.llm_client = llm_client

        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "50"))
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))

    def answer(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()

        candidates = self.retriever.retrieve(
            query=query,
            top_k=self.retrieval_top_k,
        )

        ranked_contexts = self.reranker.rerank(
            query=query,
            passages=candidates,
            top_k=self.rerank_top_k,
        )

        answer = self.llm_client.generate_answer(
            query=query,
            contexts=ranked_contexts,
        )

        return {
            "query": query,
            "answer": answer,
            "sources": ranked_contexts,
            "metadata": {
                "retrieval_top_k": self.retrieval_top_k,
                "rerank_top_k": self.rerank_top_k,
                "num_candidates": len(candidates),
                "num_contexts": len(ranked_contexts),
            },
        }


rag_client = RAGClient()