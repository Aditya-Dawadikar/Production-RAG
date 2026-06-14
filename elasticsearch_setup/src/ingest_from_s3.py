import os
import tempfile
from pathlib import Path

import boto3
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "wiki_chunks")

S3_BUCKET = os.getenv("S3_BUCKET", "prod-rag-bucket")
S3_PREFIX = os.getenv("S3_PREFIX", "wiki-chunks")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))

es = Elasticsearch(ES_URL)
s3 = boto3.client("s3")


def list_parquet_files(bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")

    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                files.append(key)

    return files


def actions_from_df(df: pd.DataFrame):
    for row in df.itertuples(index=False):
        yield {
            "_index": ES_INDEX,
            "_id": row.chunk_id,
            "_source": {
                "chunk_id": row.chunk_id,
                "doc_id": row.doc_id,
                "source_file": row.source_file,
                "chunk_index": int(row.chunk_index),
                "chunk_text": row.chunk_text,
            },
        }


def main():
    parquet_keys = list_parquet_files(S3_BUCKET, S3_PREFIX)

    print(f"Found {len(parquet_keys)} parquet files")

    total_docs = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for key in tqdm(parquet_keys):
            local_path = tmpdir / Path(key).name

            s3.download_file(S3_BUCKET, key, str(local_path))

            df = pd.read_parquet(local_path)

            helpers.bulk(
                es,
                actions_from_df(df),
                chunk_size=BATCH_SIZE,
                request_timeout=120,
            )

            total_docs += len(df)

            local_path.unlink()

    es.indices.refresh(index=ES_INDEX)

    print(f"Indexed {total_docs} chunks into {ES_INDEX}")


if __name__ == "__main__":
    main()