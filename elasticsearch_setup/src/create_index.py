import os
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "wiki_chunks")

es = Elasticsearch(ES_URL)

mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "wiki_text_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "source_file": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "chunk_text": {
                "type": "text",
                "analyzer": "wiki_text_analyzer"
            }
        }
    }
}

if es.indices.exists(index=ES_INDEX):
    print(f"Index already exists: {ES_INDEX}")
else:
    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"Created index: {ES_INDEX}")