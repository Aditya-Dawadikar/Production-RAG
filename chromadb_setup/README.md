# ChromaDB Setup

Initialize and manage Chroma vector database from S3-stored embeddings and data.

## Overview

This module provides functionality to:
- **Restore** an existing Chroma DB from S3
- **Rebuild** a Chroma DB from embedding parquet files stored on S3

## Prerequisites

- Python 3.8+
- AWS credentials configured (for S3 access)
- Access to S3 bucket containing embeddings and/or Chroma DB backup

## Installation

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Required Packages

- **Data Processing**: pandas, pyarrow
- **AWS**: boto3, s3fs
- **Vector DB**: chromadb
- **Embeddings**: sentence-transformers, torch
- **Utilities**: tqdm

## Configuration

### Environment Variables (Optional)

Set AWS credentials if not using default profile:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-west-2
```

### S3 Paths

- **Embeddings parquet files**: `s3://your-bucket/path/to/embeddings/`
- **Chroma DB backup**: `s3://your-bucket/path/to/chroma_db/`

## Usage

### Mode 1: Rebuild from Embeddings

Build a new Chroma collection from embedding parquet files:

```bash
python src/init_chroma_from_s3.py \
  --mode rebuild_from_embeddings \
  --embeddings_s3_path s3://prod-rag-bucket/wiki-embeddings/ \
  --chroma_dir ./chroma_db \
  --collection wiki \
  --batch_size 1000
```

**Parameters:**
- `--embeddings_s3_path`: S3 path to parquet files with embeddings
- `--chroma_dir`: Local directory for Chroma DB persistence
- `--collection`: Collection name to create/upsert
- `--batch_size`: Number of records per batch (default: 1000)

**Expected Parquet Columns:**
- `chunk_id`: Unique identifier for each chunk
- `chunk_text`: Text content
- `embedding`: Vector embedding array

### Mode 2: Restore from Backup

Restore an existing Chroma DB from S3:

```bash
python src/init_chroma_from_s3.py \
  --mode restore_existing \
  --bucket prod-rag-bucket \
  --chroma_s3_prefix wiki-chroma-backup/ \
  --chroma_dir ./chroma_db \
  --collection wiki
```

**Parameters:**
- `--bucket`: S3 bucket name
- `--chroma_s3_prefix`: S3 prefix where Chroma DB is stored
- `--chroma_dir`: Local directory to restore to
- `--collection`: Collection name to verify

## Verification

### Check Collection Count

```bash
python -c "
import chromadb
client = chromadb.PersistentClient('./chroma_db')
collection = client.get_collection('wiki')
print(f'Collection: wiki, Count: {collection.count()}')
"
```

### Check Disk Usage

```bash
du -sh ./chroma_db
```

## Sync to S3 (Post-Build)

After building, backup the Chroma DB to S3:

```bash
aws s3 sync ./chroma_db s3://prod-rag-bucket/wiki-chroma-backup/
```

## Troubleshooting

- **Missing AWS credentials**: Configure with `aws configure` or set environment variables
- **S3 access denied**: Verify bucket name and IAM permissions
- **Missing columns in parquet**: Ensure embeddings file has `chunk_id`, `chunk_text`, and `embedding` columns
- **Out of memory**: Reduce `--batch_size` parameter

## Notes

- Chroma DB files are persisted locally to the specified `--chroma_dir`
- Large embeddings files may require multiple batch iterations
- Progress is tracked and printed to console during ingestion
