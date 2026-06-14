# Prod RAG Service

A production-style Retrieval-Augmented Generation (RAG) service exposed via FastAPI.
It answers questions over a Wikipedia chunk corpus using hybrid retrieval
(ChromaDB semantic search + Elasticsearch keyword search), a FlashRank
cross-encoder reranker, and an LLM (Groq, via LangChain) for final answer
generation.

## Architecture

```
                 ┌────────────────────┐
   query  ─────▶ │   QueryProcessor    │  normalize + embed (sentence-transformers)
                 └─────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                            ▼
      ┌───────────────┐           ┌─────────────────┐
      │   ChromaDB     │           │  Elasticsearch   │
      │  (semantic /   │           │  (BM25 keyword)  │
      │   HNSW index)  │           │                  │
      └───────┬────────┘           └────────┬─────────┘
              │                              │
              └──────────────┬──────────────┘
                              ▼
                     merge + dedupe (HybridRetriever)
                              │
                              ▼
                     ┌──────────────────┐
                     │  FlashRank        │  cross-encoder rerank
                     │  reranker         │  -> top-N contexts
                     └─────────┬─────────┘
                                │
                                ▼
                     ┌──────────────────┐
                     │  Groq LLM         │  LangChain ChatGroq
                     │  (rag.md prompt)  │  -> final answer
                     └─────────┬─────────┘
                                │
                                ▼
                          InferenceResponse
```

## Project structure

```
prod-rag/
├── src/
│   ├── main.py                # FastAPI app: /health, /inference
│   ├── rag.py                  # RAGClient: orchestrates the pipeline
│   ├── query_processor.py      # normalizes + embeds the query
│   ├── chromadb_client.py       # ChromaDB (HNSW) semantic retrieval
│   ├── elasticsearch_client.py  # Elasticsearch BM25 retrieval
│   ├── retriever.py            # HybridRetriever: merges & dedupes results
│   ├── reranker.py              # FlashRank reranker
│   ├── llm_client.py            # ChatGroq + prompt loading
│   ├── prompts/rag.md           # RAG prompt template
│   └── data_models/Inference.py # Pydantic request/response models
├── setup/                       # EC2 provisioning & data population scripts
├── evals/                       # evaluation dataset + eval runner (WIP)
├── tests/                       # (planned)
├── requirements.txt
└── .env.example
```

## Configuration

All configuration is read from environment variables (via `.env`, see
`.env.example`).

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-transformers model used to embed queries |
| `CHROMA_DIR` | `./local/chroma` | Local path to the persistent Chroma DB |
| `CHROMA_COLLECTION` | `wiki_chunks` | Chroma collection name |
| `CHROMA_HNSW_CONSTRUCTION_EF` | `200` | HNSW `construction_ef` (index build quality, set at collection creation) |
| `CHROMA_HNSW_SEARCH_EF` | `100` | HNSW `search_ef` (recall vs. speed at query time, set at collection creation) |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint |
| `ELASTICSEARCH_INDEX` | `wiki_chunks` | Elasticsearch index name |
| `GROQ_API_KEY` | _(required)_ | API key for Groq |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq chat model used for answer generation |
| `LLM_TEMPERATURE` | `0` | LLM sampling temperature |
| `RAG_PROMPT` | `rag` | Prompt template name (file in `src/prompts/`) |
| `RETRIEVAL_TOP_K` | `50` | Candidates pulled from each retriever before reranking |
| `RERANK_TOP_K` | `5` | Contexts kept after reranking and sent to the LLM |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-12-v2` | FlashRank model name |
| `S3_BUCKET` | `prod-rag-bucket` | S3 bucket used by setup scripts for data population |
| `ES_INGEST_S3_PREFIX` | `wiki-chunks` | S3 prefix containing Elasticsearch ingestion parquet files |
| `ES_INGEST_BATCH_SIZE` | `1000` | Bulk batch size for Elasticsearch ingestion |
| `CHROMA_BACKUP_S3_PREFIX` | `wiki-chroma-backup` | S3 prefix containing the Chroma DB backup |

> **Note on HNSW params:** `hnsw:construction_ef` and `hnsw:search_ef` only take
> effect when a Chroma collection is first created. Restoring an existing
> backup from S3 keeps whatever HNSW settings it was originally built with —
> to apply new values you need to rebuild the collection (see
> `chromadb_setup/src/init_chroma_from_s3.py --mode rebuild_from_embeddings`).

`.env` is git-ignored — never commit real secrets (`GROQ_API_KEY`, AWS keys).
Only `.env.example` (with placeholder values) is tracked.

## EC2 deployment requirements

### Instance specs

Minimum recommended: **`t3.large`** (2 vCPU / 8 GB RAM), Ubuntu 22.04/24.04 LTS.

- Elasticsearch heap is fixed at `-Xms1g -Xmx1g` by `setup_elasticsearch.sh`;
  ES also wants roughly that much again free for filesystem cache.
- The FastAPI process loads the `sentence-transformers` embedding model and
  the FlashRank reranker model into memory (~1 GB combined).
- Chroma's HNSW index is loaded in memory; its size scales with corpus size
  and `CHROMA_HNSW_*` params — size up for larger corpora.

**Storage**: at least 30 GB gp3 EBS (OS, apt packages, venv, HF model cache,
Elasticsearch data dir, Chroma persistent dir). Increase to match the size of
the restored Chroma DB / Elasticsearch index.

### Security group

| Port | Protocol | Source | Purpose |
| --- | --- | --- | --- |
| 22 | TCP | your IP / bastion only | SSH admin access |
| 8000 | TCP | clients / load balancer | FastAPI (`uvicorn`) |
| 9200 | TCP | self (security group only) | Elasticsearch — `setup_elasticsearch.sh` disables `xpack.security`, so this port must **never** be exposed to `0.0.0.0/0` |

**Outbound**: allow `443`/`80` to the internet — required for `apt` package
installs, S3 access, Hugging Face model downloads (sentence-transformers /
FlashRank), and the Groq API.

### IAM permissions

Attach an IAM instance profile granting read access to `S3_BUCKET`, used by
`setup/ingest_es_from_s3.py` and `setup/restore_chroma_from_s3.py`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::prod-rag-bucket"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::prod-rag-bucket/*"
    }
  ]
}
```

`boto3` picks these up automatically from the instance profile — no AWS
access keys need to be set in `.env`.

## Quick start (EC2 / fresh Ubuntu host)

```bash
git clone <repo-url>
cd prod-rag
cp .env.example .env   # fill in GROQ_API_KEY, S3_BUCKET, etc.
bash setup/setup_ec2.sh
```

`setup_ec2.sh` is the single entry point. It runs, in order:

1. **`setup_ubuntu.sh`** – apt update/upgrade, installs Java, Python, build
   tools; creates a `venv/` and installs `requirements.txt`.
2. **`setup_elasticsearch.sh`** – installs and configures a single-node
   Elasticsearch 8.x cluster, then creates the index and ingests data from S3
   (`create_es_index.py`, `ingest_es_from_s3.py`).
3. **`setup_chromadb.sh`** – restores the Chroma vector store from its S3
   backup (`restore_chroma_from_s3.py`).

All steps are idempotent — re-running `setup_ec2.sh` skips work that's
already done (existing ES service, populated index, populated Chroma
collection). Each stage prints color-coded, staged logs.

Once setup finishes, start the API:

```bash
source venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Local development

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

You'll need a running Elasticsearch instance (see `elasticsearch_setup/`) and
a populated Chroma DB at `CHROMA_DIR` (see `chromadb_setup/`). Then run:

```bash
uvicorn src.main:app --reload
```

## API

### `GET /health`

Returns `{"status": "healthy"}`.

### `POST /inference`

Request:

```json
{ "query": "What is the capital of France?" }
```

Response:

```json
{
  "query": "What is the capital of France?",
  "answer": "...",
  "sources": [ { "id": "...", "text": "...", "metadata": {...}, "score": 0.0, "...": "..." } ],
  "metadata": {
    "retrieval_top_k": 50,
    "rerank_top_k": 5,
    "num_candidates": 50,
    "num_contexts": 5
  }
}
```

Interactive docs are available at `/docs` (Swagger) and `/redoc`.

## Evals & tests

`evals/` and `tests/` are scaffolded but not yet implemented.
