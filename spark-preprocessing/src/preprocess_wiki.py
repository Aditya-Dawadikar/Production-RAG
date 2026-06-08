import argparse
import re
from typing import List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, input_file_name, monotonically_increasing_id, sha2, concat_ws, udf
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

def clean_text(text:str) -> str:
    if text is None:
        return ""
    
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text

def chunk_text(text:str,
               chunk_size: int,
               chunk_overlap: int,
               min_chunk_words: int) -> List[Tuple[int, str]]:
    if not text:
        return []
    
    words = text.split()

    if len(words) == 0:
        return []
    
    chunks = []
    start = 0
    chunk_index = 0
    step = chunk_size - chunk_overlap
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()

        if len(chunk) > min_chunk_words:
            chunks.append((chunk_index, chunk))
        
        chunk_index += 1
        start += step
    
    return chunks

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Wikipedia text files into RAG chunks.")

    parser.add_argument("--input", required=True, help="Input Wikipedia text folder.")
    parser.add_argument("--output", required=True, help="Output Parquet path.")
    parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Number of overlapping words.")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional file limit for local testing.")
    parser.add_argument("--min-chunk-words", type=int, default=20, help="Minimum number of words required to keep a chunk.")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    
    # Create the Spark Driver
    # Driver builds the execution plan
    # Executors do the actual distributed work
    spark = (
        SparkSession.builder
        .appName("wiki-spark-preprocessing")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    chunk_schema = ArrayType(
        StructType(
            [
                StructField("chunk_index", IntegerType(), nullable=False),
                StructField("chunk_text", StringType(), nullable=False),
            ]
        )
    )

    clean_text_udf = udf(clean_text, StringType()) # User Defined Function - UDF
    chunk_text_udf = udf(lambda text: chunk_text(
                                        text=text,
                                        chunk_size=args.chunk_size,
                                        chunk_overlap=args.chunk_overlap,
                                        min_chunk_words=args.min_chunk_words,
                                    ),
                            chunk_schema
                        )
    
    """
        Raw Dataframe Columns: [raw_text, source_file]   
    """
    raw_df = (
        spark.read
            .option("wholetext", True) # one file = one row
            .text(args.input)
            .withColumnRenamed("value", "raw_text")
            .withColumn("source_file", input_file_name())
    )

    if args.limit_files:
        raw_df = raw_df.limit(args.limit_files)
    
    """
        Cleaned Text Dataframe columns: clean_text, doct_id
    """
    non_empty_df = (
        raw_df
        .withColumn("clean_text", clean_text_udf(col("raw_text")))
        .filter(col("clean_text") != "")
    )

    cleaned_df = (
        non_empty_df
        .withColumn("doc_id", sha2(concat_ws("||", col("source_file"), col("clean_text")), 256))
        .dropDuplicates(["doc_id"])
    )

    """
        Chunked Text Dataframe columns: chunks, chunk, chunk_id

        Before explode:
        ```
            one row = one document with many chunks
        ```

        After explode:
        ```
        one row = one chunk
        ```
        
        So this:
        ```
        doc_1 → [chunk_0, chunk_1, chunk_2]
        ```
        
        becomes:
        ```
        doc_1, chunk_0
        doc_1, chunk_1
        doc_1, chunk_2
        ```

        This is the core transformation for RAG.
    """
    chunked_df = (
        cleaned_df
        .withColumn("chunks", chunk_text_udf(col("clean_text")))
        .withColumn("chunk", explode(col("chunks")))
        .select(
            col("doc_id"),
            col("source_file"),
            col("chunk.chunk_index").alias("chunk_index"),
            col("chunk.chunk_text").alias("chunk_text"),
        )
        .withColumn("chunk_id", sha2(concat_ws("||", col("doc_id"), col("chunk_index")), 256))
    )

    # Select Final Columns
    final_df = (
        chunked_df
        .select(
            "chunk_id",
            "doc_id",
            "source_file",
            "chunk_index",
            "chunk_text",
        )
    )

    cleaned_df.select("source_file", "clean_text").show(20, truncate=80)

    final_df.select("source_file", "chunk_index", "chunk_text").show(20, truncate=100)

    # Write chunks to parquet file
    final_df.write.mode("overwrite").parquet(args.output)

    raw_count = raw_df.count()
    clean_count = cleaned_df.count()
    chunk_count = final_df.count()

    print("========== PREPROCESSING SUMMARY ==========")
    print(f"Raw files read: {raw_count}")
    print(f"Non-empty documents: {non_empty_count}")
    print(f"Deduplicated documents: {deduped_count}")
    print(f"Chunks generated: {chunk_count}")
    print(f"Output written to: {args.output}")

    spark.stop()

if __name__ == "__main__":
    main()

"""
### Run Command:
```
python src/jobs/preprocess_wiki.py ^
  --input data/input/plain-text-wikipedia-simpleenglish/1of2 ^
  --output data/output/processed/wiki_chunks.parquet ^
  --chunk-size 200 ^
  --chunk-overlap 40 ^
  --limit-files 20
```
"""