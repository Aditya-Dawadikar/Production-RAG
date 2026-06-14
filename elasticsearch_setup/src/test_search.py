import os
import sys
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "wiki_chunks")

query = " ".join(sys.argv[1:])

es = Elasticsearch(ES_URL)

res = es.search(
    index=ES_INDEX,
    body={
        "size": 5,
        "query": {
            "match": {
                "chunk_text": query
            }
        }
    }
)

for hit in res["hits"]["hits"]:
    src = hit["_source"]
    print("=" * 80)
    print("score:", hit["_score"])
    print("chunk_id:", src["chunk_id"])
    print("source:", src["source_file"])
    print("chunk_index:", src["chunk_index"])
    print(src["chunk_text"][:500])