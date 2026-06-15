"""Generate the eval Q&A dataset by sampling passages from the raw wiki corpus.

One-time / occasional script: run manually when the corpus changes
significantly. Output is committed to git so run_evals.py always evaluates
against a fixed, comparable benchmark set. Only needs GROQ_API_KEY - no
Chroma/Elasticsearch connection required.

Usage:
    python evals/generate_dataset.py
"""

import json
import os
import random
import re
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

from src.llm_client import llm_client  # noqa: E402

load_dotenv(PROD_RAG_ROOT / ".env")

EVAL_SAMPLE_SIZE = int(os.getenv("EVAL_SAMPLE_SIZE", "20"))
EVAL_DATASET_PATH = PROD_RAG_ROOT / os.getenv(
    "EVAL_DATASET_PATH", "evals/dataset/wiki_eval_dataset.json"
)
EVAL_WIKI_DATASET_DIR = (
    PROD_RAG_ROOT
    / os.getenv("EVAL_WIKI_DATASET_DIR", "../wiki_dataset/plain-text-wikipedia-simpleenglish")
).resolve()

PASSAGE_WORDS = 200
CORPUS_SUBDIRS = ("1of2", "2of2")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def list_corpus_files() -> list[Path]:
    files = []

    for subdir in CORPUS_SUBDIRS:
        files.extend(sorted((EVAL_WIKI_DATASET_DIR / subdir).iterdir()))

    return files


def sample_passage(file_path: Path) -> str | None:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    words = clean_text(raw_text).split()

    if len(words) < PASSAGE_WORDS:
        return None

    start = random.randint(0, len(words) - PASSAGE_WORDS)

    return " ".join(words[start : start + PASSAGE_WORDS])


def parse_qa_response(raw_output: str) -> dict | None:
    text = raw_output.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[len("json") :]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    question = str(data.get("question", "")).strip()
    reference = str(data.get("reference", "")).strip()

    if not question or not reference:
        return None

    return {"question": question, "reference": reference}


def build_qa_chain():
    prompt_text = llm_client._load_prompt("eval_qa_gen")
    prompt = PromptTemplate(template=prompt_text, input_variables=["passage"])

    return prompt | llm_client.llm | StrOutputParser()


def main():
    corpus_files = list_corpus_files()
    random.shuffle(corpus_files)

    chain = build_qa_chain()

    dataset = []

    for file_path in corpus_files:
        if len(dataset) >= EVAL_SAMPLE_SIZE:
            break

        passage = sample_passage(file_path)
        if passage is None:
            continue

        source_file = "/".join(file_path.relative_to(EVAL_WIKI_DATASET_DIR).parts)

        raw_output = chain.invoke({"passage": passage})
        qa_pair = parse_qa_response(raw_output)

        if qa_pair is None:
            print(f"  skip {source_file}: unparsable LLM output")
            continue

        dataset.append({**qa_pair, "source_file": source_file})
        print(f"  [{len(dataset)}/{EVAL_SAMPLE_SIZE}] {source_file}: {qa_pair['question']}")

    if len(dataset) < EVAL_SAMPLE_SIZE:
        print(
            f"Warning: generated only {len(dataset)}/{EVAL_SAMPLE_SIZE} Q&A pairs "
            f"(corpus exhausted after {len(corpus_files)} files)."
        )

    EVAL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(dataset)} Q&A pairs to {EVAL_DATASET_PATH}")


if __name__ == "__main__":
    main()
