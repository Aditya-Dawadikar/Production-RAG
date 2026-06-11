import argparse
import os
from pathlib import Path

import boto3
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/") + "/"


def list_s3_parquet_files(bucket: str, prefix: str):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    prefix = normalize_prefix(prefix)
    files = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Important: prevents wiki-chunks-test from matching wiki-chunks
            if not key.startswith(prefix):
                continue

            if key.endswith(".parquet"):
                files.append(key)

    return sorted(files)


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def output_key(output_prefix: str, input_key: str) -> str:
    output_prefix = normalize_prefix(output_prefix)
    name = Path(input_key).name
    return f"{output_prefix}embedding-{name}"


def output_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3")

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket", required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=512)

    args = parser.parse_args()

    input_prefix = normalize_prefix(args.input_prefix)
    output_prefix = normalize_prefix(args.output_prefix)

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown")

    print("=" * 80, flush=True)
    print(f"WORKER STARTED | worker_id={args.worker_id} | gpu={gpu_id}", flush=True)
    print(f"Input prefix:  s3://{args.bucket}/{input_prefix}", flush=True)
    print(f"Output prefix: s3://{args.bucket}/{output_prefix}", flush=True)
    print("=" * 80, flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check GPU drivers / PyTorch CUDA install.")

    print(f"CUDA devices visible: {torch.cuda.device_count()}", flush=True)
    print(f"Loading model: {args.model_name}", flush=True)

    model = SentenceTransformer(args.model_name, device="cuda")

    print("Model loaded on GPU", flush=True)

    all_files = list_s3_parquet_files(args.bucket, input_prefix)
    assigned_files = all_files[args.worker_id::args.num_workers]

    print(f"Total parquet files under exact prefix: {len(all_files)}", flush=True)
    print(f"Assigned files: {len(assigned_files)}", flush=True)

    for file_index, input_key in enumerate(assigned_files, start=1):
        out_key = output_key(output_prefix, input_key)

        print("-" * 80, flush=True)
        print(f"Worker {args.worker_id}: file {file_index}/{len(assigned_files)}", flush=True)
        print(f"Input:  s3://{args.bucket}/{input_key}", flush=True)
        print(f"Output: s3://{args.bucket}/{out_key}", flush=True)

        if output_exists(args.bucket, out_key):
            print("Output already exists. Skipping.", flush=True)
            continue

        df = pd.read_parquet(s3_uri(args.bucket, input_key))

        if "chunk_text" not in df.columns:
            raise ValueError(f"Missing chunk_text column in {input_key}")

        df = df[df["chunk_text"].notna()]
        df = df[df["chunk_text"].astype(str).str.strip() != ""]
        df = df.copy()

        total_rows = len(df)
        print(f"Rows to embed: {total_rows}", flush=True)

        if total_rows == 0:
            df["embedding"] = []
            df.to_parquet(s3_uri(args.bucket, out_key), index=False, engine="pyarrow")
            print(f"Completed empty file: s3://{args.bucket}/{out_key}", flush=True)
            continue

        embeddings = []
        texts = df["chunk_text"].astype(str).tolist()

        for start in range(0, total_rows, args.batch_size):
            end = min(start + args.batch_size, total_rows)
            batch_texts = texts[start:end]

            batch_embeddings = model.encode(
                batch_texts,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            embeddings.extend(batch_embeddings.astype("float32").tolist())

            print(
                f"Progress: {end}/{total_rows} rows ({(end / total_rows) * 100:.2f}%)",
                flush=True,
            )

        df["embedding"] = embeddings

        df.to_parquet(
            s3_uri(args.bucket, out_key),
            index=False,
            engine="pyarrow",
        )

        print(f"Completed: s3://{args.bucket}/{out_key}", flush=True)

    print("=" * 80, flush=True)
    print(f"WORKER COMPLETE | worker_id={args.worker_id}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()