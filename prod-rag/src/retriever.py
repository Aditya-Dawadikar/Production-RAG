from typing import Any

from src.chromadb_client import chromadb_client
from src.elasticsearch_client import elasticsearch_client
from src.query_processor import query_processor


class HybridRetriever:
    def __init__(self):
        self.chroma = chromadb_client
        self.elasticsearch = elasticsearch_client
        self.query_processor = query_processor

    def retrieve(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
        processed_query = self.query_processor.process(query)

        semantic_results = self.chroma.retrieve(
            query_embedding=processed_query["embedding"],
            query_text=processed_query["query"],
            top_k=top_k,
        )

        keyword_results = self.elasticsearch.retrieve(
            query=processed_query["query"],
            top_k=top_k,
        )

        merged_results = self._merge_and_dedupe(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
        )

        return merged_results[:top_k]

    def _merge_and_dedupe(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen = {}

        for result in semantic_results:
            chunk_id = result["id"]

            seen[chunk_id] = {
                **result,
                "semantic_score": result.get("score", 0.0),
                "keyword_score": 0.0,
                "retrieval_sources": ["chroma"],
            }

        for result in keyword_results:
            chunk_id = result["id"]

            if chunk_id in seen:
                seen[chunk_id]["keyword_score"] = result.get("score", 0.0)
                seen[chunk_id]["retrieval_sources"].append("elasticsearch")
            else:
                seen[chunk_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "keyword_score": result.get("score", 0.0),
                    "retrieval_sources": ["elasticsearch"],
                }

        results = list(seen.values())

        results.sort(
            key=lambda x: (
                len(x["retrieval_sources"]),
                x["semantic_score"],
                x["keyword_score"],
            ),
            reverse=True,
        )

        return results


retriever = HybridRetriever()