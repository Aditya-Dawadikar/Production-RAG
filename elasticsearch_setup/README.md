# Elasticsearch Setup

Install and configure Elasticsearch for full-text search on wiki chunks, then ingest data from S3.

## Overview

This module provides a complete pipeline for:
- **Installing** Elasticsearch 8.x on Linux with optimized JVM settings
- **Creating** an index with text analysis and metadata fields
- **Ingesting** preprocessed wiki chunks from S3 parquet files
- **Testing** search functionality

## Prerequisites

- Linux server (Ubuntu 20.04+)
- Java 21 (installed via script)
- Python 3.10+
- AWS credentials configured (for S3 access)
- Access to S3 bucket containing parquet files

## Installation

### Step 1: Install Elasticsearch

Run the installation script on your EC2/Linux server:

```bash
bash src/install_elasticsearch.sh
```

This script will:
- Install Java 21 and dependencies
- Add Elasticsearch 8.x repository
- Install Elasticsearch service
- Disable security (suitable for internal networks)
- Configure heap size (1GB)
- Enable and start the service
- Verify Elasticsearch is running

**Configuration Details:**
- Cluster: `rag-es`
- Node: `rag-es-node-1`
- Port: 9200
- Discovery: Single-node
- Replicas: 0 (disabled)

### Step 2: Set up Python Environment

```bash
bash src/setup_python.sh
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

```bash
export ES_URL="http://localhost:9200"
export ES_INDEX="wiki_chunks"
export S3_BUCKET="prod-rag-bucket"
export S3_PREFIX="wiki-chunks"
export BATCH_SIZE="1000"
```

## Usage

### Create Index

Initialize the Elasticsearch index with schema and text analyzer:

```bash
source venv/bin/activate
python src/create_index.py
```

**Index Schema:**
- `chunk_id` (keyword): Unique identifier
- `doc_id` (keyword): Document identifier
- `source_file` (keyword): Source file path
- `chunk_index` (integer): Chunk position in document
- `chunk_text` (text): Searchable text with English stopword filtering

### Ingest Data from S3

Load parquet files from S3 into Elasticsearch:

```bash
source venv/bin/activate
python src/ingest_from_s3.py
```

**Environment Variables for Ingestion:**
- `ES_URL`: Elasticsearch endpoint (default: http://localhost:9200)
- `ES_INDEX`: Index name (default: wiki_chunks)
- `S3_BUCKET`: S3 bucket containing parquet files
- `S3_PREFIX`: S3 prefix/path to parquet files (default: wiki-chunks)
- `BATCH_SIZE`: Records per bulk insert (default: 1000)

**Process:**
1. Lists all `.parquet` files in S3 prefix
2. Downloads each file to temp directory
3. Reads parquet data into pandas DataFrame
4. Bulk inserts documents into Elasticsearch
5. Refreshes index for immediate search availability
6. Cleans up temp files

### Test Search

Query the index and display top 5 results:

```bash
source venv/bin/activate
python src/test_search.py "your search query here"
```

Example:

```bash
python src/test_search.py "machine learning algorithms"
```

**Output:**
- Relevance score
- Chunk ID
- Source file
- Chunk index
- First 500 characters of text

## Monitoring

### Check Index Status

```bash
curl http://localhost:9200/_cat/indices
```

### Get Index Settings

```bash
curl http://localhost:9200/wiki_chunks
```

### Get Document Count

```bash
curl http://localhost:9200/wiki_chunks/_count
```

### View Node Info

```bash
curl http://localhost:9200
```

### Check Service Status

```bash
sudo systemctl status elasticsearch
```

## Troubleshooting

### Elasticsearch won't start

Check logs:
```bash
sudo tail -f /var/log/elasticsearch/rag-es.log
```

Check service:
```bash
sudo systemctl status elasticsearch --no-pager -l
```

### Memory issues

Adjust heap size in `/etc/elasticsearch/jvm.options.d/heap.options`:
```
-Xms512m
-Xmx512m
```

Then restart:
```bash
sudo systemctl restart elasticsearch
```

### S3 access errors

Verify AWS credentials:
```bash
aws s3 ls s3://your-bucket/
```

### Connection refused

Verify Elasticsearch is running:
```bash
curl http://localhost:9200
```

Check firewall/security groups allow port 9200.

## Performance Tips

- **Reduce replica count**: Already set to 0
- **Batch size tuning**: Increase `BATCH_SIZE` for larger batches (default 1000)
- **Request timeout**: Set to 120 seconds for large ingestions
- **Refresh**: Index is refreshed once after all ingestion completes

## Data Format

Parquet files must contain these columns:
- `chunk_id`: String, unique identifier
- `doc_id`: String, document identifier
- `source_file`: String, source file path
- `chunk_index`: Integer, position in document
- `chunk_text`: String, text content to be indexed

## Notes

- Security is disabled (xpack.security: false) - suitable for internal networks only
- Uses standard text analyzer with English stopwords
- Single-node cluster configuration
- No data replication (number_of_replicas: 0)
- Bulk API used for efficient ingestion
