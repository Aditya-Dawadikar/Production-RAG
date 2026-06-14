import argparse
import os
from pathlib import Path

import boto3
import chromadb
import pandas as pd


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


def restore_existing_chroma_db(args):
    download_s3_prefix(
        bucket=args.bucket,
        prefix=args.chroma_s3_prefix,
        local_dir=args.chroma_dir,
    )

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.collection)

    print("Chroma DB restored.")
    print(f"Collection: {args.collection}")
    print(f"Count: {collection.count()}")


def rebuild_chroma_from_embedding_parquet(args):
    client = chromadb.PersistentClient(path=args.chroma_dir)

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": args.hnsw_construction_ef,
            "hnsw:search_ef": args.hnsw_search_ef,
        },
    )

    df = pd.read_parquet(args.embeddings_s3_path)

    required_cols = {"chunk_id", "chunk_text", "embedding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["chunk_id", "chunk_text", "embedding"]).copy()
    df["chunk_id"] = df["chunk_id"].astype(str)

    total = len(df)
    print(f"Rows loaded: {total}")

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch = df.iloc[start:end].copy()

        ids = batch["chunk_id"].astype(str).tolist()
        documents = batch["chunk_text"].astype(str).tolist()

        embeddings = [
            emb.tolist() if hasattr(emb, "tolist") else emb
            for emb in batch["embedding"].tolist()
        ]

        metadata_cols = [
            col for col in batch.columns
            if col not in {"chunk_text", "embedding"}
        ]

        metadatas = (
            batch[metadata_cols]
            .fillna("")
            .astype(str)
            .to_dict("records")
        )

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"Inserted {end}/{total}")

    print("Chroma DB rebuilt.")
    print(f"Count: {collection.count()}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["restore_db", "rebuild_from_embeddings"],
        required=True,
    )

    parser.add_argument("--bucket")
    parser.add_argument("--chroma_s3_prefix")

    parser.add_argument("--embeddings_s3_path")

    parser.add_argument("--chroma_dir", default="./chroma_db")
    parser.add_argument("--collection", default="wiki")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hnsw_construction_ef", type=int, default=200)
    parser.add_argument("--hnsw_search_ef", type=int, default=100)

    args = parser.parse_args()

    if args.mode == "restore_db":
        if not args.bucket or not args.chroma_s3_prefix:
            raise ValueError("--bucket and --chroma_s3_prefix are required")

        restore_existing_chroma_db(args)

    elif args.mode == "rebuild_from_embeddings":
        if not args.embeddings_s3_path:
            raise ValueError("--embeddings_s3_path is required")

        rebuild_chroma_from_embedding_parquet(args)


if __name__ == "__main__":
    main()