# Production RAG

An end-to-end pipeline for building and serving a production-style
Retrieval-Augmented Generation (RAG) system over a Wikipedia (Simple English)
corpus: data prep → embedding → indexing → hybrid retrieval + reranking + LLM
answer generation, with evals.

## Pipeline overview

```
wiki_dataset/            raw Wikipedia (Simple English) corpus (Kaggle, not in git)
        │
        ▼
spark-preprocessing/     chunk articles with Spark, write parquet to S3
        │
        ▼
embedding/                embed chunks (sentence-transformers, GPU), write parquet to S3
        │
        ├──────────────────────────────┐
        ▼                                ▼
chromadb_setup/                  elasticsearch_setup/
build/restore Chroma vector       build ES index + ingest
store from S3 embeddings           chunks from S3
        │                                │
        └───────────────┬───────────────┘
                          ▼
                  prod-rag/
                  FastAPI RAG service: hybrid retrieval
                  (Chroma + Elasticsearch) → FlashRank
                  reranker → Groq LLM → answer
                  (+ evals/ for retrieval & answer quality)
```

## Components

| Directory | Purpose |
| --- | --- |
| [`wiki_dataset/`](wiki_dataset/README.md) | Raw corpus download instructions (not tracked in git) |
| [`spark-preprocessing/`](spark-preprocessing/README.md) | Spark jobs that chunk the raw wiki articles |
| [`embedding/`](embedding/local/notes.md) | GPU embedding worker that turns chunks into vectors |
| [`chromadb_setup/`](chromadb_setup/README.md) | Build/restore the Chroma vector store from S3 |
| [`elasticsearch_setup/`](elasticsearch_setup/README.md) | Install Elasticsearch and ingest chunks for keyword search |
| [`prod-rag/`](prod-rag/README.md) | The FastAPI RAG service, EC2 deployment, and evals |

## Getting started

The serving component is [`prod-rag/`](prod-rag/README.md) — see its README
for architecture, configuration, EC2 deployment, local development, the API,
and evals. The other directories are the offline pipeline used to produce the
data that `prod-rag` serves (raw corpus → chunks → embeddings → Chroma /
Elasticsearch).
