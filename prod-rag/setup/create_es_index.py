# prod-rag/setup/create_es_index.py
#
# Creates the Elasticsearch index used by src/elasticsearch_client.py,
# matching the schema produced by elasticsearch_setup/src/create_index.py.

import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "wiki_chunks")

es = Elasticsearch(ES_URL)

mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "wiki_text_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_",
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
                "analyzer": "wiki_text_analyzer",
            },
        }
    },
}

if es.indices.exists(index=ES_INDEX):
    print(f"Index already exists: {ES_INDEX}")
else:
    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"Created index: {ES_INDEX}")
