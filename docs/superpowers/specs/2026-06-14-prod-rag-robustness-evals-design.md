# prod-rag Robustness Evals: Design

## Goal

Extend the `prod-rag` eval framework (see
[2026-06-14-prod-rag-evals-design.md](2026-06-14-prod-rag-evals-design.md))
with two new test categories that probe the failure modes `src/prompts/rag.md`
explicitly tries to avoid ("If the context does not contain enough
information: say that you do not know. Do not invent facts."):

- **`unanswerable`** — questions with no relevant source in the corpus at
  all. Retrieval should find nothing useful, and the model should say it
  doesn't know rather than answer anyway.
- **`hallucination_trap`** — questions about a topic that *is* in the
  corpus, asking for a specific detail (date/number/name/award/location)
  that the retrieved passage does **not** state. Retrieval will plausibly
  return on-topic context; the risk is the model fabricates the missing
  detail instead of saying it isn't stated.

Both categories are scored on the same axis: **did the model avoid
fabricating a confident, unsupported answer?** This reuses the same
LLM-as-judge model (`GROQ_MODEL`) and the existing eval runner, per the
"Same `run_evals.py`, combined report" decision.

## File layout

```
prod-rag/
├── evals/
│   ├── dataset/
│   │   ├── wiki_eval_dataset.json             # existing - answerable golden set
│   │   ├── wiki_eval_unanswerable.json        # NEW - hand-curated, out-of-corpus questions
│   │   └── wiki_eval_hallucination_trap.json  # NEW - LLM-generated near-miss questions
│   ├── corpus_utils.py                        # NEW - shared passage-sampling/JSON-parsing helpers
│   ├── generate_dataset.py                    # MODIFIED - import shared helpers from corpus_utils
│   ├── generate_hallucination_dataset.py      # NEW - generates wiki_eval_hallucination_trap.json
│   ├── run_evals.py                           # MODIFIED - category-aware, combined report
│   ├── requirements.txt                       # unchanged - openevals already provides HALLUCINATION_PROMPT
│   └── results/
├── src/
│   └── prompts/
│       ├── eval_qa_gen.md                     # existing
│       └── eval_hallucination_qa_gen.md       # NEW - near-miss question generation prompt
├── .env / .env.example                        # add EVAL_UNANSWERABLE_DATASET_PATH,
│                                               #     EVAL_HALLUCINATION_DATASET_PATH,
│                                               #     EVAL_HALLUCINATION_SAMPLE_SIZE
└── README.md                                  # document new categories/scripts/config
```

## `evals/dataset/wiki_eval_unanswerable.json` (new, hand-curated)

8 questions, no `reference`/`source_file` (none apply — nothing in the
corpus answers these):

```json
[
  {"question": "What is the current price of Bitcoin in US dollars?"},
  {"question": "What is tomorrow's weather forecast for Tokyo?"},
  {"question": "Who won the most recent season of American Idol?"},
  {"question": "What is your favorite movie?"},
  {"question": "How do I reset the password for my email account?"},
  {"question": "What new AI models were released in 2025?"},
  {"question": "What is the phone number for the nearest pizza restaurant?"},
  {"question": "Which team won the Super Bowl this year?"}
]
```

Committed to git as a fixed set, like `wiki_eval_dataset.json`. No
generation script needed.

## `evals/corpus_utils.py` (new — refactor)

`generate_dataset.py` and the new `generate_hallucination_dataset.py` both
need the same passage-sampling and JSON-parsing logic. Extract it into a
shared module (pure refactor of existing code, no behavior change):

```python
import json
import random
import re
from pathlib import Path


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def list_corpus_files(wiki_dataset_dir: Path, subdirs: tuple[str, ...]) -> list[Path]:
    files = []
    for subdir in subdirs:
        files.extend(sorted((wiki_dataset_dir / subdir).iterdir()))
    return files


def sample_passage(file_path: Path, passage_words: int) -> str | None:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    words = clean_text(raw_text).split()

    if len(words) < passage_words:
        return None

    start = random.randint(0, len(words) - passage_words)
    return " ".join(words[start : start + passage_words])


def parse_json_response(raw_output: str) -> dict | None:
    text = raw_output.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[len("json"):]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
```

Both `generate_dataset.py` and `generate_hallucination_dataset.py` run as
scripts (`python evals/generate_*.py`), so Python puts `evals/` on
`sys.path[0]` automatically — `import corpus_utils` works with no extra path
setup.

### `generate_dataset.py` changes

- Remove the local `clean_text`, `list_corpus_files`, `sample_passage`
  definitions; import them from `corpus_utils`.
- `list_corpus_files()` call becomes
  `list_corpus_files(EVAL_WIKI_DATASET_DIR, CORPUS_SUBDIRS)`.
- `sample_passage(file_path)` call becomes
  `sample_passage(file_path, PASSAGE_WORDS)`.
- `parse_qa_response` becomes a thin wrapper around the shared
  `parse_json_response`:

```python
def parse_qa_response(raw_output: str) -> dict | None:
    data = parse_json_response(raw_output)
    if data is None:
        return None

    question = str(data.get("question", "")).strip()
    reference = str(data.get("reference", "")).strip()

    if not question or not reference:
        return None

    return {"question": question, "reference": reference}
```

No change to `generate_dataset.py`'s CLI, env vars, or output format.

## `src/prompts/eval_hallucination_qa_gen.md` (new)

```markdown
You are creating evaluation data for a retrieval-augmented generation (RAG) system, specifically to test whether it correctly admits uncertainty instead of fabricating facts.

Read the passage below and:
- Identify the main entity or topic the passage is about, and note the specific facts (dates, numbers, names, places, awards, etc.) the passage states about it.
- Write one factual-sounding question about that same entity or topic, asking for a specific detail (a date, number, name, award, or location) that is plausible to ask about but is NOT stated anywhere in the passage.
- Briefly describe, in a few words, what specific detail the question asks for that is missing from the passage.

Respond with a single JSON object and nothing else - no markdown, no code fences, no extra commentary - in exactly this format:
{{"question": "<question text>", "missing_detail": "<short description of the missing detail>"}}

Passage:
{passage}
```

(Double braces escape the literal JSON braces for `PromptTemplate`'s
f-string formatting, same as `eval_qa_gen.md`.)

## `evals/generate_hallucination_dataset.py` (new)

Same structure as `generate_dataset.py`: shuffle corpus files, sample a
200-word passage, invoke the LLM, parse the response, repeat until
`EVAL_HALLUCINATION_SAMPLE_SIZE` items are collected or the corpus is
exhausted (warn if under target, same as `generate_dataset.py`).

```python
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
```

Each generated item has the shape:
```json
{"question": "...", "missing_detail": "...", "source_file": "1of2/wiki_NN"}
```
`missing_detail` is for human review/debugging only — it documents what the
"trap" is but is not used in scoring.

## `evals/run_evals.py` changes

### New scoring path (no `reference` available)

For `unanswerable` and `hallucination_trap` items there is no ground-truth
answer, so the reference-dependent metrics
(`LLMContextPrecisionWithReference`, `LLMContextRecall`, `correctness`)
don't apply. Instead:

- **New constant**: `ROBUSTNESS_RAGAS_METRICS = [Faithfulness()]` —
  `Faithfulness` only needs `user_input`/`response`/`retrieved_contexts`, no
  `reference`.
- **New OpenEvals judge**, added in `build_openevals_judges()`:
  ```python
  from openevals.prompts import HALLUCINATION_PROMPT  # add to existing import

  hallucination = create_llm_as_judge(
      prompt=HALLUCINATION_PROMPT,
      judge=llm_client.llm,
      feedback_key="hallucination",
      continuous=True,
  )
  ```
  `HALLUCINATION_PROMPT`'s rubric explicitly rewards "appropriately
  indicates uncertainty when information is incomplete" and penalizes
  unsupported dates/numbers/names — directly measuring the
  fabricate-vs-hedge behavior these categories test.
- `build_openevals_judges()` now returns 3 values
  (`retrieval_relevance, correctness, hallucination`); its call site in
  `main()` (`retrieval_relevance, correctness = build_openevals_judges()`)
  is updated to unpack all three.
- **Existing `retrieval_relevance` judge is reused as-is** — for
  `unanswerable` it's expected to score low (nothing relevant retrieved);
  for `hallucination_trap` it's expected to score moderate/high (right
  topic, missing specific fact). Kept as a diagnostic signal alongside
  `hallucination`.

**`evaluate_ragas` is modified to take a `metrics` parameter** (was
hardcoded to the global `RAGAS_METRICS`), so it can be reused for both
metric sets:

```python
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
```

New function, parallel to the existing `evaluate_openevals`:

```python
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
```

`reference_outputs=""` is required because `HALLUCINATION_PROMPT` contains a
`{reference_outputs}` placeholder — `create_llm_as_judge` only includes
template variables that are not `None`, and `str.format()` raises `KeyError`
on a referenced-but-missing key, so an explicit empty string must be passed.

### `run_single` becomes category-aware

```python
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
```

`evaluate_openevals` is unchanged from the existing implementation.

### `main()` becomes category-driven

```python
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
```

For each category in `CATEGORIES`:
1. Load its dataset file (reusing `load_dataset`, parameterized by path),
   apply `--limit` (per category — `--limit 2` runs 2 items from *each* of
   the 3 categories, i.e. 6 `rag_client.answer()` calls total for a smoke
   test).
2. Run `run_single(item, category, ...)` for each item, with the existing
   per-item try/except error isolation.
3. Print a console table for that category (`print(f"\n=== {category}
   ===")` then the existing `print_console_report`, using that category's
   `metric_names`).

After all categories, write **one combined JSON report**:

```json
{
  "metadata": { "...": "... (unchanged fields, plus dataset_paths per category)" },
  "categories": {
    "answerable": {"results": [...], "means": {...}},
    "unanswerable": {"results": [...], "means": {...}},
    "hallucination_trap": {"results": [...], "means": {...}}
  }
}
```

`metadata.dataset_paths` becomes `{"answerable": "...", "unanswerable":
"...", "hallucination_trap": "..."}` (replacing the single
`metadata.dataset_path` string). Written to the same
`EVAL_RESULTS_DIR/<timestamp>.json` path as before.

## Config additions

Add to `.env`, `.env.example`, and the README config table:

| Variable | Default | Description |
| --- | --- | --- |
| `EVAL_UNANSWERABLE_DATASET_PATH` | `evals/dataset/wiki_eval_unanswerable.json` | Path to the hand-curated out-of-corpus eval questions |
| `EVAL_HALLUCINATION_DATASET_PATH` | `evals/dataset/wiki_eval_hallucination_trap.json` | Path to the generated hallucination-trap eval questions |
| `EVAL_HALLUCINATION_SAMPLE_SIZE` | `10` | Number of questions `generate_hallucination_dataset.py` produces |

No new packages — `openevals` (already in `evals/requirements.txt`)
provides `HALLUCINATION_PROMPT`.

## README updates

- Project structure tree: add `src/prompts/eval_hallucination_qa_gen.md`,
  `evals/corpus_utils.py`, `evals/generate_hallucination_dataset.py`,
  `evals/dataset/wiki_eval_unanswerable.json`,
  `evals/dataset/wiki_eval_hallucination_trap.json`.
- Configuration table: add the 3 new rows above.
- "Evals & tests" section: describe all three categories (`answerable`,
  `unanswerable`, `hallucination_trap`), the new generation script
  (`python evals/generate_hallucination_dataset.py`), and the robustness
  metric set (`faithfulness`, `retrieval_relevance`, `hallucination`)
  alongside the existing answerable-category metrics.

## Testing / validation

- `generate_hallucination_dataset.py`: same validation approach as
  `generate_dataset.py` — run once, inspect
  `wiki_eval_hallucination_trap.json` for `EVAL_HALLUCINATION_SAMPLE_SIZE`
  well-formed entries with non-empty `question`/`missing_detail`/
  `source_file`. Only needs `GROQ_API_KEY`.
- `run_evals.py`: validate with `--limit 2` first (6 total
  `rag_client.answer()` calls across the 3 categories), check the console
  prints 3 separate tables with correct per-category metric columns and
  `MEAN` rows, and the combined JSON report has a `categories` key with all
  three category names. Then run the full set. Same environment
  requirements as before (populated Chroma, reachable Elasticsearch,
  `GROQ_API_KEY`).
- Spot-check: for `unanswerable` items, `retrieval_relevance` scores should
  generally be low and `hallucination` scores should generally be high
  (model correctly says it doesn't know). For `hallucination_trap` items,
  `retrieval_relevance` may be moderate/high while `hallucination` reveals
  whether the model invented the missing detail anyway. These are
  observations to look at in the report, not asserted thresholds (no
  pass/fail gating, consistent with the existing eval runner).

## Out of scope / future work

- No pass/fail thresholds or CI gating — purely observational, same as the
  existing eval runner.
- `wiki_eval_unanswerable.json` is a fixed hand-curated list; growing it is
  a manual future edit, not automated.
- Re-running `generate_hallucination_dataset.py` to refresh the
  hallucination-trap set is a manual, occasional operation (like
  `generate_dataset.py`).
