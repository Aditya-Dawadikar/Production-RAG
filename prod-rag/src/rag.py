# src/rag.py

import hashlib
import logging
import os
from typing import Any

from dotenv import load_dotenv

from src.llm_client import llm_client
from src.reranker import reranker
from src.retriever import retriever

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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
        query_digest = _digest(query)

        logger.info("Pipeline start | query_sha256=%s query_len=%d", query_digest, len(query))
        logger.debug("Pipeline start | query=%r", query)

        logger.info("[1/3] Retrieval | top_k=%d", self.retrieval_top_k)
        candidates = self.retriever.retrieve(
            query=query,
            top_k=self.retrieval_top_k,
        )
        logger.info("[1/3] Retrieval done | candidates=%d", len(candidates))
        logger.debug(
            "[1/3] Retrieval done | ids=%s",
            [c.get("id") for c in candidates],
        )

        logger.info("[2/3] Reranking | top_k=%d", self.rerank_top_k)
        ranked_contexts = self.reranker.rerank(
            query=query,
            passages=candidates,
            top_k=self.rerank_top_k,
        )
        logger.info(
            "[2/3] Reranking done | contexts=%d scores=%s",
            len(ranked_contexts),
            [round(c.get("score", 0.0), 4) for c in ranked_contexts],
        )
        logger.debug(
            "[2/3] Reranking done | ids=%s",
            [c.get("id") for c in ranked_contexts],
        )

        logger.info("[3/3] LLM generation | model=%s", self.llm_client.model_name)
        answer = self.llm_client.generate_answer(
            query=query,
            contexts=ranked_contexts,
        )
        logger.info(
            "[3/3] LLM generation done | answer_sha256=%s answer_len=%d",
            _digest(answer),
            len(answer),
        )
        logger.debug("[3/3] LLM generation done | answer=%r", answer)

        result = {
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

        logger.info("Pipeline done | query_sha256=%s", query_digest)

        return result


rag_client = RAGClient()