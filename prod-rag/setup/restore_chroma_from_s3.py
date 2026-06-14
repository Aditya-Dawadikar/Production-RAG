# prod-rag/setup/restore_chroma_from_s3.py
#
# Restores the local Chroma persistent directory from an S3 backup,
# trimmed from chromadb_setup/src/init_chroma_from_s3.py (restore_db mode).
# Skips the restore if the collection is already populated.

import os
from pathlib import Path

import boto3
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./local/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "wiki_chunks")

S3_BUCKET = os.getenv("S3_BUCKET", "prod-rag-bucket")
S3_PREFIX = os.getenv("CHROMA_BACKUP_S3_PREFIX", "wiki-chroma-backup")


def download_s3_prefix(bucket: str, prefix: str, local_dir: str):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    local_root = Path(local_dir).resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.endswith("/"):
                continue

            relative_path = key[len(prefix):].lstrip("/")
            local_path = (local_root / relative_path).resolve()

            if not local_path.is_relative_to(local_root):
                raise ValueError(f"Refusing to restore outside chroma dir: {key}")

            local_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading s3://{bucket}/{key} -> {local_path}")
            s3.download_file(bucket, key, str(local_path))


def existing_count() -> int | None:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(CHROMA_COLLECTION)
        return collection.count()
    except Exception:
        return None


def main():
    count = existing_count()

    if count:
        print(f"Collection '{CHROMA_COLLECTION}' already has {count} vectors. Skipping restore.")
        return

    print(f"Restoring Chroma DB from s3://{S3_BUCKET}/{S3_PREFIX} -> {CHROMA_DIR}")

    download_s3_prefix(
        bucket=S3_BUCKET,
        prefix=S3_PREFIX,
        local_dir=CHROMA_DIR,
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(CHROMA_COLLECTION)

    print("Chroma DB restored.")
    print(f"Collection: {CHROMA_COLLECTION}")
    print(f"Count: {collection.count()}")


if __name__ == "__main__":
    main()
