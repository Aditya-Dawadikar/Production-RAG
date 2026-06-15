"""Generate "hallucination trap" questions by sampling passages from the raw wiki corpus.

Each question asks for a specific detail about an in-corpus topic that the
sampled passage does not state, to test whether the RAG pipeline correctly
says it doesn't know rather than fabricating the detail.

One-time / occasional script, like generate_dataset.py. Only needs
GROQ_API_KEY - no Chroma/Elasticsearch connection required.

Usage:
    python evals/generate_hallucination_dataset.py
"""

import json
import os
import random
import sys
from pathlib import Path

PROD_RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_RAG_ROOT))

# Corpus passages can contain non-ASCII text that the default Windows
# console encoding (cp1252) can't print.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402

from corpus_utils import list_corpus_files, parse_json_response, sample_passage  # noqa: E402
from src.llm_client import llm_client  # noqa: E402

load_dotenv(PROD_RAG_ROOT / ".env")

EVAL_HALLUCINATION_SAMPLE_SIZE = int(os.getenv("EVAL_HALLUCINATION_SAMPLE_SIZE", "10"))
EVAL_HALLUCINATION_DATASET_PATH = PROD_RAG_ROOT / os.getenv(
    "EVAL_HALLUCINATION_DATASET_PATH", "evals/dataset/wiki_eval_hallucination_trap.json"
)
EVAL_WIKI_DATASET_DIR = (
    PROD_RAG_ROOT
    / os.getenv("EVAL_WIKI_DATASET_DIR", "../wiki_dataset/plain-text-wikipedia-simpleenglish")
).resolve()

PASSAGE_WORDS = 200
CORPUS_SUBDIRS = ("1of2", "2of2")


def parse_hallucination_response(raw_output: str) -> dict | None:
    data = parse_json_response(raw_output)
    if data is None:
        return None

    question = str(data.get("question", "")).strip()
    missing_detail = str(data.get("missing_detail", "")).strip()

    if not question or not missing_detail:
        return None

    return {"question": question, "missing_detail": missing_detail}


def build_hallucination_chain():
    prompt_text = llm_client._load_prompt("eval_hallucination_qa_gen")
    prompt = PromptTemplate(template=prompt_text, input_variables=["passage"])

    return prompt | llm_client.llm | StrOutputParser()


def main():
    corpus_files = list_corpus_files(EVAL_WIKI_DATASET_DIR, CORPUS_SUBDIRS)
    random.shuffle(corpus_files)

    chain = build_hallucination_chain()

    dataset = []

    for file_path in corpus_files:
        if len(dataset) >= EVAL_HALLUCINATION_SAMPLE_SIZE:
            break

        passage = sample_passage(file_path, PASSAGE_WORDS)
        if passage is None:
            continue

        source_file = "/".join(file_path.relative_to(EVAL_WIKI_DATASET_DIR).parts)

        raw_output = chain.invoke({"passage": passage})
        qa_pair = parse_hallucination_response(raw_output)

        if qa_pair is None:
            print(f"  skip {source_file}: unparsable LLM output")
            continue

        dataset.append({**qa_pair, "source_file": source_file})
        print(f"  [{len(dataset)}/{EVAL_HALLUCINATION_SAMPLE_SIZE}] {source_file}: {qa_pair['question']}")

    if len(dataset) < EVAL_HALLUCINATION_SAMPLE_SIZE:
        print(
            f"Warning: generated only {len(dataset)}/{EVAL_HALLUCINATION_SAMPLE_SIZE} "
            f"hallucination-trap questions (corpus exhausted after {len(corpus_files)} files)."
        )

    EVAL_HALLUCINATION_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EVAL_HALLUCINATION_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(dataset)} hallucination-trap questions to {EVAL_HALLUCINATION_DATASET_PATH}")


if __name__ == "__main__":
    main()
