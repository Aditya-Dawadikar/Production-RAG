import argparse
import os
import time
from typing import Iterator

import pandas as pd
from pyspark import TaskContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, input_file_name, trim
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


MODEL = None
MODEL_NAME = None
BATCH_SIZE = 64


def log(message: str):
    print(f"\n========== {message} ==========\n", flush=True)


def executor_log(message: str):
    context = TaskContext.get()

    partition_id = context.partitionId() if context else "unknown"
    attempt_id = context.attemptNumber() if context else "unknown"

    executor_id = os.environ.get("SPARK_EXECUTOR_ID", "local")

    print(
        f"[EXECUTOR={executor_id} | PARTITION={partition_id} | ATTEMPT={attempt_id}] "
        f"{message}",
        flush=True,
    )


def embed_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    global MODEL, MODEL_NAME, BATCH_SIZE

    if MODEL is None:
        executor_log(f"Loading model: {MODEL_NAME}")

        from sentence_transformers import SentenceTransformer

        start_time = time.time()
        MODEL = SentenceTransformer(MODEL_NAME)
        load_time = time.time() - start_time

        executor_log(f"Model loaded in {load_time:.2f}s")

    processed_rows = 0
    batch_id = 0

    for pdf in iterator:
        batch_id += 1
        pdf = pdf.copy()

        rows_in_batch = len(pdf)

        executor_log(
            f"Starting pandas batch={batch_id}, rows={rows_in_batch}"
        )

        texts = pdf["chunk_text"].fillna("").astype(str).tolist()

        if not texts:
            executor_log(f"Skipping empty batch={batch_id}")
            continue

        start_time = time.time()

        embeddings = MODEL.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        encode_time = time.time() - start_time

        pdf["embedding"] = embeddings.astype("float32").tolist()

        processed_rows += rows_in_batch

        executor_log(
            f"Finished batch={batch_id}, "
            f"rows={rows_in_batch}, "
            f"partition_rows_done={processed_rows}, "
            f"encode_time={encode_time:.2f}s"
        )

        yield pdf

    executor_log(f"Partition complete. Total rows embedded={processed_rows}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for Wikipedia RAG chunks."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--arrow-batch-size", type=int, default=256)
    parser.add_argument("--aws-region", type=str, default="us-west-2")

    parser.add_argument(
        "--output-partitions",
        type=int,
        default=2,
        help="Number of Spark partitions used for embedding generation.",
    )

    return parser.parse_args()


def create_spark_session(
    aws_region: str,
    arrow_batch_size: int,
    output_partitions: int,
) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("wiki-embedding-generation")
        .master(f"local[{output_partitions}]")
        .config("spark.driver.memory", "10g")
        .config("spark.sql.shuffle.partitions", str(output_partitions))
        .config("spark.default.parallelism", str(output_partitions))
        .config("spark.local.dir", "/tmp")
        .config("spark.ui.host", "0.0.0.0")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config(
            "spark.sql.execution.arrow.maxRecordsPerBatch",
            str(arrow_batch_size),
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.endpoint.region", aws_region)
        .getOrCreate()
    )

    return spark


def main():
    global MODEL_NAME, BATCH_SIZE

    args = parse_args()

    MODEL_NAME = args.model_name
    BATCH_SIZE = args.batch_size

    log("STARTING EMBEDDING JOB")

    spark = create_spark_session(
        aws_region=args.aws_region,
        arrow_batch_size=args.arrow_batch_size,
        output_partitions=args.output_partitions,
    )

    spark.sparkContext.setLogLevel("WARN")

    input_schema = StructType([
        StructField("chunk_id", StringType(), False),
        StructField("doc_id", StringType(), True),
        StructField("source_file", StringType(), True),
        StructField("chunk_index", IntegerType(), True),
        StructField("chunk_text", StringType(), True),
    ])

    output_schema = StructType([
        StructField("chunk_id", StringType(), False),
        StructField("doc_id", StringType(), True),
        StructField("source_file", StringType(), True),
        StructField("chunk_index", IntegerType(), True),
        StructField("chunk_text", StringType(), True),
        StructField("embedding", ArrayType(FloatType()), False),
    ])

    log("READING INPUT PARQUET FROM S3")

    df = (
        spark.read
        .schema(input_schema)
        .parquet(args.input)
        .withColumn("_input_file", input_file_name())
    )

    filtered_df = (
        df
        .filter(col("chunk_text").isNotNull())
        .filter(trim(col("chunk_text")) != "")
    )

    log("COUNTING INPUT ROWS AND FILES")

    filtered_df.cache()

    total_rows = filtered_df.count()
    total_files = filtered_df.select("_input_file").distinct().count()

    log(
        f"INPUT SUMMARY\n"
        f"Input path: {args.input}\n"
        f"Input files: {total_files}\n"
        f"Rows to embed: {total_rows}\n"
        f"Output partitions: {args.output_partitions}\n"
        f"Approx rows per partition: {total_rows // args.output_partitions}"
    )

    repartitioned_df = (
        filtered_df
        .drop("_input_file")
        .repartition(args.output_partitions)
    )

    spark.sparkContext.setJobGroup(
        "wiki-embedding-generation",
        "Generate embeddings from chunk parquet files",
    )

    log("STARTING EMBEDDING + S3 WRITE")

    job_start_time = time.time()

    embedded_df = repartitioned_df.mapInPandas(
        embed_partition,
        schema=output_schema,
    )

    embedded_df.write.mode("overwrite").parquet(args.output)

    job_time = time.time() - job_start_time

    log("EMBEDDING JOB COMPLETE")

    print("========== EMBEDDING SUMMARY ==========")
    print(f"Input path: {args.input}")
    print(f"Output path: {args.output}")
    print(f"Model name: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Arrow batch size: {args.arrow_batch_size}")
    print(f"Input files: {total_files}")
    print(f"Rows embedded: {total_rows}")
    print(f"Output partitions: {args.output_partitions}")
    print(f"Embedding + write time: {job_time:.2f}s")
    print(f"Embeddings written to: {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()