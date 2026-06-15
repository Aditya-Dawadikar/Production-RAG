"""Run retrieval/generation evals for the prod-rag pipeline.

Requires a populated Chroma collection, a reachable Elasticsearch, and a
valid GROQ_API_KEY - the same environment as running the FastAPI server.
Purely observational: no pass/fail gating.

Usage:
    python evals/run_evals.py [--limit N]
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROD_RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROD_RAG_ROOT))

# Dataset/answer text can contain non-ASCII content that the default Windows
# console encoding (cp1252) can't print.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from openevals.llm import create_llm_as_judge  # noqa: E402
from openevals.prompts import (  # noqa: E402
    CORRECTNESS_PROMPT,
    HALLUCINATION_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from src.llm_client import llm_client  # noqa: E402
from src.rag import rag_client  # noqa: E402

load_dotenv(PROD_RAG_ROOT / ".env")

EVAL_DATASET_PATH = PROD_RAG_ROOT / os.getenv(
    "EVAL_DATASET_PATH", "evals/dataset/wiki_eval_dataset.json"
)
EVAL_UNANSWERABLE_DATASET_PATH = PROD_RAG_ROOT / os.getenv(
    "EVAL_UNANSWERABLE_DATASET_PATH", "evals/dataset/wiki_eval_unanswerable.json"
)
EVAL_HALLUCINATION_DATASET_PATH = PROD_RAG_ROOT / os.getenv(
    "EVAL_HALLUCINATION_DATASET_PATH", "evals/dataset/wiki_eval_hallucination_trap.json"
)
EVAL_RESULTS_DIR = PROD_RAG_ROOT / os.getenv("EVAL_RESULTS_DIR", "evals/results")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

RAGAS_METRICS = [
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    Faithfulness(),
    ResponseRelevancy(),
]

ROBUSTNESS_RAGAS_METRICS = [Faithfulness()]

CATEGORIES = {
    "answerable": {
        "dataset_path": EVAL_DATASET_PATH,
        "metric_names": [metric.name for metric in RAGAS_METRICS] + ["retrieval_relevance", "correctness"],
    },
    "unanswerable": {
        "dataset_path": EVAL_UNANSWERABLE_DATASET_PATH,
        "metric_names": [metric.name for metric in ROBUSTNESS_RAGAS_METRICS] + ["retrieval_relevance", "hallucination"],
    },
    "hallucination_trap": {
        "dataset_path": EVAL_HALLUCINATION_DATASET_PATH,
        "metric_names": [metric.name for metric in ROBUSTNESS_RAGAS_METRICS] + ["retrieval_relevance", "hallucination"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval/generation evals for prod-rag.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N dataset items per category."
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path, limit: int | None) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if limit is not None:
        items = items[:limit]

    return items


def build_ragas_judges():
    evaluator_llm = LangchainLLMWrapper(llm_client.llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    )

    return evaluator_llm, evaluator_embeddings


def build_openevals_judges():
    retrieval_relevance = create_llm_as_judge(
        prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
        judge=llm_client.llm,
        feedback_key="retrieval_relevance",
        continuous=True,
    )
    correctness = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=llm_client.llm,
        feedback_key="correctness",
        continuous=True,
    )
    hallucination = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        judge=llm_client.llm,
        feedback_key="hallucination",
        continuous=True,
    )

    return retrieval_relevance, correctness, hallucination


def evaluate_ragas(sample, metrics, evaluator_llm, evaluator_embeddings) -> dict:
    dataset = EvaluationDataset(samples=[sample])

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    row = result.to_pandas().iloc[0]

    return {metric.name: float(row[metric.name]) for metric in metrics}


def evaluate_openevals(item, answer, retrieved_contexts, retrieval_relevance, correctness) -> dict:
    context_text = "\n\n".join(retrieved_contexts)

    relevance_result = retrieval_relevance(inputs=item["question"], context=context_text)
    correctness_result = correctness(
        inputs=item["question"],
        outputs=answer,
        reference_outputs=item["reference"],
    )

    return {
        "retrieval_relevance": float(relevance_result["score"]),
        "correctness": float(correctness_result["score"]),
    }


def evaluate_robustness_openevals(item, answer, retrieved_contexts, retrieval_relevance, hallucination) -> dict:
    context_text = "\n\n".join(retrieved_contexts)

    relevance_result = retrieval_relevance(inputs=item["question"], context=context_text)
    hallucination_result = hallucination(
        inputs=item["question"], outputs=answer, context=context_text, reference_outputs="",
    )

    return {
        "retrieval_relevance": float(relevance_result["score"]),
        "hallucination": float(hallucination_result["score"]),
    }


def run_single(item, category, evaluator_llm, evaluator_embeddings, retrieval_relevance, correctness, hallucination) -> dict:
    pipeline_result = rag_client.answer(item["question"])

    answer = pipeline_result["answer"]
    retrieved_contexts = [source["text"] for source in pipeline_result["sources"]]

    if category == "answerable":
        sample = SingleTurnSample(
            user_input=item["question"],
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=item["reference"],
        )
        scores = evaluate_ragas(sample, RAGAS_METRICS, evaluator_llm, evaluator_embeddings)
        scores.update(
            evaluate_openevals(item, answer, retrieved_contexts, retrieval_relevance, correctness)
        )
    else:
        sample = SingleTurnSample(
            user_input=item["question"],
            response=answer,
            retrieved_contexts=retrieved_contexts,
        )
        scores = evaluate_ragas(sample, ROBUSTNESS_RAGAS_METRICS, evaluator_llm, evaluator_embeddings)
        scores.update(
            evaluate_robustness_openevals(item, answer, retrieved_contexts, retrieval_relevance, hallucination)
        )

    return {
        "question": item["question"],
        "reference": item.get("reference"),
        "source_file": item.get("source_file"),
        "answer": answer,
        "retrieved_contexts": retrieved_contexts,
        "scores": scores,
    }


def print_console_report(records: list[dict], metric_names: list[str]) -> None:
    rows = []

    for record in records:
        row = {"question": record["question"][:60]}

        if "error" in record:
            row["error"] = record["error"][:60]
            for metric in metric_names:
                row[metric] = None
        else:
            row["error"] = ""
            for metric in metric_names:
                row[metric] = round(record["scores"][metric], 4)

        rows.append(row)

    mean_row = {"question": "MEAN", "error": ""}
    for metric in metric_names:
        values = [record["scores"][metric] for record in records if "error" not in record]
        mean_row[metric] = round(sum(values) / len(values), 4) if values else None
    rows.append(mean_row)

    print(pd.DataFrame(rows).to_string(index=False))


def run_category(category, evaluator_llm, evaluator_embeddings, retrieval_relevance, correctness, hallucination, limit) -> dict:
    config = CATEGORIES[category]
    dataset = load_dataset(config["dataset_path"], limit)

    print(f"\n=== {category} ===")
    print(f"Loaded {len(dataset)} eval items from {config['dataset_path']}")

    records = []

    for idx, item in enumerate(dataset, start=1):
        print(f"[{idx}/{len(dataset)}] {item['question'][:80]}")

        try:
            record = run_single(
                item, category, evaluator_llm, evaluator_embeddings,
                retrieval_relevance, correctness, hallucination,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            record = {
                "question": item["question"],
                "reference": item.get("reference"),
                "source_file": item.get("source_file"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

        records.append(record)

    print()
    print_console_report(records, config["metric_names"])

    means = {}
    for metric in config["metric_names"]:
        values = [record["scores"][metric] for record in records if "error" not in record]
        means[metric] = sum(values) / len(values) if values else None

    return {"results": records, "means": means}


def main():
    args = parse_args()

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    evaluator_llm, evaluator_embeddings = build_ragas_judges()
    retrieval_relevance, correctness, hallucination = build_openevals_judges()

    categories_report = {}

    for category in CATEGORIES:
        categories_report[category] = run_category(
            category, evaluator_llm, evaluator_embeddings,
            retrieval_relevance, correctness, hallucination, args.limit,
        )

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_paths": {
                category: str(config["dataset_path"]) for category, config in CATEGORIES.items()
            },
            "groq_model": llm_client.model_name,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "retrieval_top_k": rag_client.retrieval_top_k,
            "rerank_top_k": rag_client.rerank_top_k,
        },
        "categories": categories_report,
    }

    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = EVAL_RESULTS_DIR / f"{timestamp_slug}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nWrote report to {report_path}")


if __name__ == "__main__":
    main()
