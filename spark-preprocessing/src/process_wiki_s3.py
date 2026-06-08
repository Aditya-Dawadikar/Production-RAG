import argparse
import re
from typing import List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, explode, input_file_name, sha2, udf
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_words: int,
) -> List[Tuple[int, str]]:
    if not text:
        return []

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0
    chunk_index = 0
    step = chunk_size - chunk_overlap

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()

        if len(chunk_words) >= min_chunk_words:
            chunks.append((chunk_index, chunk))

        chunk_index += 1
        start += step

    return chunks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Wikipedia text files into RAG chunks."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--chunk-overlap", type=int, default=40)
    parser.add_argument("--min-chunk-words", type=int, default=20)

    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--aws-region", type=str, default="us-west-2")

    parser.add_argument(
        "--output-partitions",
        type=int,
        default=16,
        help="Number of output Parquet partitions/files.",
    )

    return parser.parse_args()


def create_spark_session(aws_region: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("wiki-spark-preprocessing")
        .master("local[4]")
        .config("spark.driver.memory", "10g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", "16")
        .config("spark.local.dir", "/tmp")
        .config("spark.ui.host", "0.0.0.0")
        .config("spark.driver.bindAddress", "0.0.0.0")
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
    args = parse_args()

    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if args.min_chunk_words <= 0:
        raise ValueError("min_chunk_words must be greater than 0")

    spark = create_spark_session(args.aws_region)
    spark.sparkContext.setLogLevel("WARN")

    chunk_schema = ArrayType(
        StructType(
            [
                StructField("chunk_index", IntegerType(), nullable=False),
                StructField("chunk_text", StringType(), nullable=False),
            ]
        )
    )

    clean_text_udf = udf(clean_text, StringType())

    chunk_text_udf = udf(
        lambda text: chunk_text(
            text=text,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_chunk_words=args.min_chunk_words,
        ),
        chunk_schema,
    )

    raw_df = (
        spark.read
        .option("wholetext", True)
        .text(args.input)
        .withColumnRenamed("value", "raw_text")
        .withColumn("source_file", input_file_name())
    )

    if args.limit_files:
        raw_df = raw_df.limit(args.limit_files)

    non_empty_df = (
        raw_df
        .withColumn("clean_text", clean_text_udf(col("raw_text")))
        .filter(col("clean_text") != "")
    )

    cleaned_df = (
        non_empty_df
        .withColumn(
            "doc_id",
            sha2(concat_ws("||", col("source_file"), col("clean_text")), 256),
        )
        .dropDuplicates(["doc_id"])
    )

    final_df = (
        cleaned_df
        .withColumn("chunks", chunk_text_udf(col("clean_text")))
        .withColumn("chunk", explode(col("chunks")))
        .select(
            col("doc_id"),
            col("source_file"),
            col("chunk.chunk_index").alias("chunk_index"),
            col("chunk.chunk_text").alias("chunk_text"),
        )
        .withColumn(
            "chunk_id",
            sha2(concat_ws("||", col("doc_id"), col("chunk_index")), 256),
        )
        .select(
            "chunk_id",
            "doc_id",
            "source_file",
            "chunk_index",
            "chunk_text",
        )
        .repartition(args.output_partitions)
    )

    final_df.cache()

    raw_count = raw_df.count()
    non_empty_count = non_empty_df.count()
    deduped_count = cleaned_df.count()
    chunk_count = final_df.count()

    final_df.write.mode("overwrite").parquet(args.output)

    print("========== PREPROCESSING SUMMARY ==========")
    print(f"Raw files read: {raw_count}")
    print(f"Non-empty documents: {non_empty_count}")
    print(f"Deduplicated documents: {deduped_count}")
    print(f"Chunks generated: {chunk_count}")
    print(f"Output written to: {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()