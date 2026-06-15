# prod-rag Robustness Evals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new eval categories to `prod-rag/evals/run_evals.py` —
`unanswerable` (out-of-corpus questions) and `hallucination_trap` (in-corpus
topic, missing-detail questions) — both scored on whether the pipeline
correctly admits it doesn't know instead of fabricating an answer, per
[2026-06-14-prod-rag-robustness-evals-design.md](../specs/2026-06-14-prod-rag-robustness-evals-design.md).

**Architecture:** Extract shared corpus-sampling/JSON-parsing helpers into a
new `evals/corpus_utils.py` module (refactor, no behavior change). Add a
hand-curated `wiki_eval_unanswerable.json` dataset and a new
`generate_hallucination_dataset.py` script (+ prompt) that produces
`wiki_eval_hallucination_trap.json`. Make `run_evals.py` category-driven:
each category has its own dataset, metric set, and console table, and all
three categories are written to one combined JSON report.

**Tech Stack:** Python, LangChain (`ChatGroq`), Ragas (`Faithfulness`,
`EvaluationDataset`, `evaluate`), OpenEvals (`create_llm_as_judge`,
`HALLUCINATION_PROMPT`, `RAG_RETRIEVAL_RELEVANCE_PROMPT`). No new
dependencies — `openevals` (already in `evals/requirements.txt`) provides
`HALLUCINATION_PROMPT`.

All commands below assume the working directory is `D:\Production_RAG\prod-rag`
(referred to as `prod-rag/` for relative paths).

---

### Task 1: Extract `corpus_utils.py` and refactor `generate_dataset.py`

**Files:**
- Create: `prod-rag/evals/corpus_utils.py`
- Modify: `prod-rag/evals/generate_dataset.py`

- [ ] **Step 1: Create `evals/corpus_utils.py`**

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

- [ ] **Step 2: Compile-check the new module**

Run: `python -m py_compile evals/corpus_utils.py`
Expected: no output (success, exit code 0).

- [ ] **Step 3: Update `generate_dataset.py` imports**

In `prod-rag/evals/generate_dataset.py`, replace the import block:

```python
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
```

with:

```python
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
```

(`import re` is removed — `clean_text` moved to `corpus_utils` and is no
longer used directly here.)

- [ ] **Step 4: Remove local helper functions, replace `parse_qa_response`**

Replace this block (the local `clean_text`, `list_corpus_files`,
`sample_passage`, and `parse_qa_response` definitions):

```python
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
```

with:

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

- [ ] **Step 5: Update call sites in `main()`**

Replace:

```python
def main():
    corpus_files = list_corpus_files()
    random.shuffle(corpus_files)
```

with:

```python
def main():
    corpus_files = list_corpus_files(EVAL_WIKI_DATASET_DIR, CORPUS_SUBDIRS)
    random.shuffle(corpus_files)
```

Replace:

```python
        passage = sample_passage(file_path)
        if passage is None:
            continue
```

with:

```python
        passage = sample_passage(file_path, PASSAGE_WORDS)
        if passage is None:
            continue
```

- [ ] **Step 6: Compile-check the refactored script**

Run: `python -m py_compile evals/generate_dataset.py`
Expected: no output (success, exit code 0).

- [ ] **Step 7: Smoke-test `corpus_utils` functions**

Run (from `prod-rag/`):

```bash
python - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "evals")
from corpus_utils import clean_text, list_corpus_files, parse_json_response, sample_passage

assert clean_text("  a   b\n\tc  ") == "a b c"

assert parse_json_response('{"a": 1}') == {"a": 1}
fence = chr(96) * 3
assert parse_json_response(f"{fence}json\n{{\"a\": 1}}\n{fence}") == {"a": 1}
assert parse_json_response("not json") is None

wiki_dir = Path("../wiki_dataset/plain-text-wikipedia-simpleenglish").resolve()
files = list_corpus_files(wiki_dir, ("1of2", "2of2"))
assert len(files) > 0

passage = sample_passage(files[0], 200)
assert passage is None or len(passage.split()) == 200

print("OK")
EOF
```

Expected output: `OK`

- [ ] **Step 8: Commit**

```bash
git add evals/corpus_utils.py evals/generate_dataset.py
git commit -m "refactor(prod-rag): extract corpus_utils helpers from generate_dataset"
```

---

### Task 2: Add hand-curated `wiki_eval_unanswerable.json`

**Files:**
- Create: `prod-rag/evals/dataset/wiki_eval_unanswerable.json`

- [ ] **Step 1: Create the dataset file**

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

- [ ] **Step 2: Validate JSON structure**

Run (from `prod-rag/`):

```bash
python - <<'EOF'
import json

with open("evals/dataset/wiki_eval_unanswerable.json", encoding="utf-8") as f:
    data = json.load(f)

assert len(data) == 8
assert all(set(item.keys()) == {"question"} and item["question"] for item in data)

print("OK")
EOF
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add evals/dataset/wiki_eval_unanswerable.json
git commit -m "feat(prod-rag): add hand-curated unanswerable eval questions"
```

---

### Task 3: Generate `wiki_eval_hallucination_trap.json`

**Files:**
- Create: `prod-rag/src/prompts/eval_hallucination_qa_gen.md`
- Create: `prod-rag/evals/generate_hallucination_dataset.py`
- Modify: `prod-rag/.env.example`
- Modify: `prod-rag/.env`
- Create (generated by script): `prod-rag/evals/dataset/wiki_eval_hallucination_trap.json`

- [ ] **Step 1: Create the prompt template**

Create `prod-rag/src/prompts/eval_hallucination_qa_gen.md`:

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

- [ ] **Step 2: Create the generation script**

Create `prod-rag/evals/generate_hallucination_dataset.py`:

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

- [ ] **Step 3: Add new config vars to `.env.example`**

In `prod-rag/.env.example`, the current eval block (end of file) is:

```
# Evals (used by evals/*.py)
EVAL_DATASET_PATH=evals/dataset/wiki_eval_dataset.json
EVAL_SAMPLE_SIZE=20
EVAL_RESULTS_DIR=evals/results
EVAL_WIKI_DATASET_DIR=../wiki_dataset/plain-text-wikipedia-simpleenglish
```

Replace it with:

```
# Evals (used by evals/*.py)
EVAL_DATASET_PATH=evals/dataset/wiki_eval_dataset.json
EVAL_SAMPLE_SIZE=20
EVAL_RESULTS_DIR=evals/results
EVAL_WIKI_DATASET_DIR=../wiki_dataset/plain-text-wikipedia-simpleenglish
EVAL_HALLUCINATION_DATASET_PATH=evals/dataset/wiki_eval_hallucination_trap.json
EVAL_HALLUCINATION_SAMPLE_SIZE=10
```

- [ ] **Step 4: Add the same vars to `.env`**

`prod-rag/.env` is git-ignored and contains real secrets — edit it directly
(don't `cat`/print it). Its eval block currently ends with:

```
EVAL_WIKI_DATASET_DIR=../wiki_dataset/plain-text-wikipedia-simpleenglish
```

Append two lines after it:

```
EVAL_HALLUCINATION_DATASET_PATH=evals/dataset/wiki_eval_hallucination_trap.json
EVAL_HALLUCINATION_SAMPLE_SIZE=10
```

- [ ] **Step 5: Compile-check the new script**

Run: `python -m py_compile evals/generate_hallucination_dataset.py`
Expected: no output (success, exit code 0).

- [ ] **Step 6: Run the generation script**

Run (from `prod-rag/`): `python evals/generate_hallucination_dataset.py`

Requires `GROQ_API_KEY` (present in `.env`), live calls to Groq, and the raw
wiki corpus at `../wiki_dataset/plain-text-wikipedia-simpleenglish`. Expected
output: 10 progress lines like
`  [1/10] 1of2/wiki_NN: <question text>` (possibly interspersed with `  skip
...: unparsable LLM output` lines), followed by
`Wrote 10 hallucination-trap questions to <path>\evals\dataset\wiki_eval_hallucination_trap.json`.

If the run produces fewer than 10 items, it prints a `Warning: generated only
N/10 ...` line — that's acceptable (matches `generate_dataset.py`'s existing
behavior), but re-run once if `N` is very low (e.g. < 5).

- [ ] **Step 7: Validate the generated dataset**

Run (from `prod-rag/`):

```bash
python - <<'EOF'
import json

with open("evals/dataset/wiki_eval_hallucination_trap.json", encoding="utf-8") as f:
    data = json.load(f)

assert len(data) >= 1
for item in data:
    assert set(item.keys()) == {"question", "missing_detail", "source_file"}
    assert item["question"] and item["missing_detail"] and item["source_file"]

print(f"OK: {len(data)} items")
EOF
```

Expected output: `OK: <N> items` (N is whatever Step 6 produced).

- [ ] **Step 8: Commit**

```bash
git add src/prompts/eval_hallucination_qa_gen.md evals/generate_hallucination_dataset.py evals/dataset/wiki_eval_hallucination_trap.json .env.example
git commit -m "feat(prod-rag): generate hallucination-trap eval dataset"
```

(`.env` is git-ignored and intentionally not staged.)

---

### Task 4: `run_evals.py` — robustness scoring helpers

**Files:**
- Modify: `prod-rag/evals/run_evals.py`

This task adds the building blocks (`ROBUSTNESS_RAGAS_METRICS`, the
`hallucination` judge, a parameterized `evaluate_ragas`, and
`evaluate_robustness_openevals`) and makes `run_single` category-aware. Task
5 wires these into a category-driven `main()`.

- [ ] **Step 1: Add `HALLUCINATION_PROMPT` to the openevals import**

Replace:

```python
from openevals.prompts import CORRECTNESS_PROMPT, RAG_RETRIEVAL_RELEVANCE_PROMPT  # noqa: E402
```

with:

```python
from openevals.prompts import (  # noqa: E402
    CORRECTNESS_PROMPT,
    HALLUCINATION_PROMPT,
    RAG_RETRIEVAL_RELEVANCE_PROMPT,
)
```

- [ ] **Step 2: Add `ROBUSTNESS_RAGAS_METRICS`**

Replace:

```python
RAGAS_METRICS = [
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    Faithfulness(),
    ResponseRelevancy(),
]
```

with:

```python
RAGAS_METRICS = [
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    Faithfulness(),
    ResponseRelevancy(),
]

ROBUSTNESS_RAGAS_METRICS = [Faithfulness()]
```

- [ ] **Step 3: Add the `hallucination` judge to `build_openevals_judges`**

Replace:

```python
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

    return retrieval_relevance, correctness
```

with:

```python
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
```

- [ ] **Step 4: Parameterize `evaluate_ragas` by `metrics`**

Replace:

```python
def evaluate_ragas(sample, evaluator_llm, evaluator_embeddings) -> dict:
    dataset = EvaluationDataset(samples=[sample])

    result = evaluate(
        dataset=dataset,
        metrics=RAGAS_METRICS,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    row = result.to_pandas().iloc[0]

    return {metric.name: float(row[metric.name]) for metric in RAGAS_METRICS}
```

with:

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

- [ ] **Step 5: Add `evaluate_robustness_openevals`**

Add this function immediately after `evaluate_openevals` (which stays
unchanged):

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

`reference_outputs=""` (empty string, not omitted) is required: `HALLUCINATION_PROMPT`
contains a `{reference_outputs}` placeholder, and `create_llm_as_judge` drops
`None`-valued params before calling `str.format()`, which would otherwise
raise `KeyError: 'reference_outputs'`.

- [ ] **Step 6: Make `run_single` category-aware**

Replace:

```python
def run_single(item, evaluator_llm, evaluator_embeddings, retrieval_relevance, correctness) -> dict:
    pipeline_result = rag_client.answer(item["question"])

    answer = pipeline_result["answer"]
    retrieved_contexts = [source["text"] for source in pipeline_result["sources"]]

    sample = SingleTurnSample(
        user_input=item["question"],
        response=answer,
        retrieved_contexts=retrieved_contexts,
        reference=item["reference"],
    )

    scores = evaluate_ragas(sample, evaluator_llm, evaluator_embeddings)
    scores.update(
        evaluate_openevals(item, answer, retrieved_contexts, retrieval_relevance, correctness)
    )

    return {
        "question": item["question"],
        "reference": item["reference"],
        "source_file": item.get("source_file"),
        "answer": answer,
        "retrieved_contexts": retrieved_contexts,
        "scores": scores,
    }
```

with:

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

- [ ] **Step 7: Compile-check**

Run: `python -m py_compile evals/run_evals.py`
Expected: no output (success, exit code 0).

Note: `main()` still calls `build_openevals_judges()` and `run_single(...)`
with their old signatures at this point — that's fine, `py_compile` doesn't
check call-site argument counts. Task 5 fixes `main()`.

- [ ] **Step 8: Commit**

```bash
git add evals/run_evals.py
git commit -m "refactor(prod-rag): add robustness scoring helpers to run_evals"
```

---

### Task 5: `run_evals.py` — category-driven `main()` and combined report

**Files:**
- Modify: `prod-rag/evals/run_evals.py`
- Modify: `prod-rag/.env.example`
- Modify: `prod-rag/.env`

- [ ] **Step 1: Add `EVAL_UNANSWERABLE_DATASET_PATH` and `EVAL_HALLUCINATION_DATASET_PATH` constants**

Replace:

```python
EVAL_DATASET_PATH = PROD_RAG_ROOT / os.getenv(
    "EVAL_DATASET_PATH", "evals/dataset/wiki_eval_dataset.json"
)
EVAL_RESULTS_DIR = PROD_RAG_ROOT / os.getenv("EVAL_RESULTS_DIR", "evals/results")
```

with:

```python
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
```

- [ ] **Step 2: Add the `CATEGORIES` dict**

After the `ROBUSTNESS_RAGAS_METRICS = [Faithfulness()]` line added in Task 4,
add:

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

- [ ] **Step 3: Update `parse_args` help text**

Replace:

```python
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N dataset items."
    )
```

with:

```python
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N dataset items per category."
    )
```

- [ ] **Step 4: Parameterize `load_dataset` by dataset path**

Replace:

```python
def load_dataset(limit: int | None) -> list[dict]:
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)

    if limit is not None:
        items = items[:limit]

    return items
```

with:

```python
def load_dataset(dataset_path: Path, limit: int | None) -> list[dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if limit is not None:
        items = items[:limit]

    return items
```

- [ ] **Step 5: Add `run_category` helper**

Add this function after `print_console_report` (and before `main`):

```python
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
```

- [ ] **Step 6: Rewrite `main()`**

Replace the entire `main()` function:

```python
def main():
    args = parse_args()

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.limit)
    print(f"Loaded {len(dataset)} eval items from {EVAL_DATASET_PATH}")

    evaluator_llm, evaluator_embeddings = build_ragas_judges()
    retrieval_relevance, correctness = build_openevals_judges()

    metric_names = [metric.name for metric in RAGAS_METRICS] + [
        "retrieval_relevance",
        "correctness",
    ]

    records = []

    for idx, item in enumerate(dataset, start=1):
        print(f"[{idx}/{len(dataset)}] {item['question'][:80]}")

        try:
            record = run_single(
                item, evaluator_llm, evaluator_embeddings, retrieval_relevance, correctness
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            record = {
                "question": item["question"],
                "reference": item["reference"],
                "source_file": item.get("source_file"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

        records.append(record)

    print()
    print_console_report(records, metric_names)

    means = {}
    for metric in metric_names:
        values = [record["scores"][metric] for record in records if "error" not in record]
        means[metric] = sum(values) / len(values) if values else None

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(EVAL_DATASET_PATH),
            "num_items": len(dataset),
            "groq_model": llm_client.model_name,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "retrieval_top_k": rag_client.retrieval_top_k,
            "rerank_top_k": rag_client.rerank_top_k,
        },
        "results": records,
        "means": means,
    }

    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = EVAL_RESULTS_DIR / f"{timestamp_slug}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nWrote report to {report_path}")
```

with:

```python
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
```

- [ ] **Step 7: Add `EVAL_UNANSWERABLE_DATASET_PATH` to `.env.example`**

In `prod-rag/.env.example`, the eval block (after Task 3) is:

```
# Evals (used by evals/*.py)
EVAL_DATASET_PATH=evals/dataset/wiki_eval_dataset.json
EVAL_SAMPLE_SIZE=20
EVAL_RESULTS_DIR=evals/results
EVAL_WIKI_DATASET_DIR=../wiki_dataset/plain-text-wikipedia-simpleenglish
EVAL_HALLUCINATION_DATASET_PATH=evals/dataset/wiki_eval_hallucination_trap.json
EVAL_HALLUCINATION_SAMPLE_SIZE=10
```

Replace it with:

```
# Evals (used by evals/*.py)
EVAL_DATASET_PATH=evals/dataset/wiki_eval_dataset.json
EVAL_SAMPLE_SIZE=20
EVAL_RESULTS_DIR=evals/results
EVAL_WIKI_DATASET_DIR=../wiki_dataset/plain-text-wikipedia-simpleenglish
EVAL_UNANSWERABLE_DATASET_PATH=evals/dataset/wiki_eval_unanswerable.json
EVAL_HALLUCINATION_DATASET_PATH=evals/dataset/wiki_eval_hallucination_trap.json
EVAL_HALLUCINATION_SAMPLE_SIZE=10
```

- [ ] **Step 8: Add the same var to `.env`**

Edit `prod-rag/.env` directly (git-ignored, don't print it). Add this line
next to the other `EVAL_*` entries:

```
EVAL_UNANSWERABLE_DATASET_PATH=evals/dataset/wiki_eval_unanswerable.json
```

- [ ] **Step 9: Compile-check**

Run: `python -m py_compile evals/run_evals.py`
Expected: no output (success, exit code 0).

- [ ] **Step 10: Manual end-to-end validation on EC2**

This script needs a populated Chroma collection, a reachable Elasticsearch,
and `GROQ_API_KEY` — only available on the EC2 deployment
(`/home/ubuntu/Production-RAG/prod-rag`, see
`chromadb_setup/local/notes.md` for SSH access). After deploying this
branch's changes there:

```bash
cd /home/ubuntu/Production-RAG/prod-rag
source venv/bin/activate
python evals/run_evals.py --limit 2
```

Expected:
- Three console sections, `=== answerable ===`, `=== unanswerable ===`,
  `=== hallucination_trap ===`, each printing a per-question table ending in
  a `MEAN` row with that category's `metric_names` as columns (`answerable`
  has `llm_context_precision_with_reference`, `context_recall`,
  `faithfulness`, `answer_relevancy`, `retrieval_relevance`, `correctness`;
  the other two have `faithfulness`, `retrieval_relevance`,
  `hallucination`).
- `Wrote report to evals/results/<timestamp>.json`.
- The written JSON has top-level `metadata.dataset_paths` (a dict with all
  three category names) and `categories` (a dict with `answerable`,
  `unanswerable`, `hallucination_trap`, each `{"results": [...], "means":
  {...}}`).

Spot-check the scores (no pass/fail gating, just sanity):
- `unanswerable` items should generally have low `retrieval_relevance` and
  high `hallucination` (model correctly says it doesn't know).
- `hallucination_trap` items should generally have moderate/high
  `retrieval_relevance`; `hallucination` reveals whether the model invented
  the missing detail.

If this looks correct, optionally run the full set (no `--limit`) and note
the result in the PR description.

- [ ] **Step 11: Commit**

```bash
git add evals/run_evals.py .env.example
git commit -m "feat(prod-rag): category-driven run_evals with combined report"
```

(`.env` is git-ignored and intentionally not staged.)

---

### Task 6: Update `README.md`

**Files:**
- Modify: `prod-rag/README.md`

- [ ] **Step 1: Update the project structure tree**

Replace:

```
├── src/
│   ├── main.py                # FastAPI app: /health, /inference
│   ├── rag.py                  # RAGClient: orchestrates the pipeline
│   ├── query_processor.py      # normalizes + embeds the query
│   ├── chromadb_client.py       # ChromaDB (HNSW) semantic retrieval
│   ├── elasticsearch_client.py  # Elasticsearch BM25 retrieval
│   ├── retriever.py            # HybridRetriever: merges & dedupes results
│   ├── reranker.py              # FlashRank reranker
│   ├── llm_client.py            # ChatGroq + prompt loading
│   ├── prompts/rag.md           # RAG prompt template
│   ├── prompts/eval_qa_gen.md   # eval dataset Q&A generation prompt
│   └── data_models/Inference.py # Pydantic request/response models
├── setup/                       # EC2 provisioning & data population scripts
├── evals/
│   ├── dataset/wiki_eval_dataset.json # eval Q&A benchmark (committed)
│   ├── generate_dataset.py      # samples corpus, generates Q&A via Groq
│   ├── run_evals.py              # Ragas + OpenEvals eval runner
│   ├── requirements.txt          # eval-only deps
│   └── results/                  # per-run JSON reports (git-ignored)
├── tests/                       # (planned)
├── requirements.txt
└── .env.example
```

with:

```
├── src/
│   ├── main.py                # FastAPI app: /health, /inference
│   ├── rag.py                  # RAGClient: orchestrates the pipeline
│   ├── query_processor.py      # normalizes + embeds the query
│   ├── chromadb_client.py       # ChromaDB (HNSW) semantic retrieval
│   ├── elasticsearch_client.py  # Elasticsearch BM25 retrieval
│   ├── retriever.py            # HybridRetriever: merges & dedupes results
│   ├── reranker.py              # FlashRank reranker
│   ├── llm_client.py            # ChatGroq + prompt loading
│   ├── prompts/rag.md           # RAG prompt template
│   ├── prompts/eval_qa_gen.md   # eval dataset Q&A generation prompt
│   ├── prompts/eval_hallucination_qa_gen.md # hallucination-trap Q&A generation prompt
│   └── data_models/Inference.py # Pydantic request/response models
├── setup/                       # EC2 provisioning & data population scripts
├── evals/
│   ├── dataset/
│   │   ├── wiki_eval_dataset.json            # answerable eval benchmark (committed)
│   │   ├── wiki_eval_unanswerable.json       # out-of-corpus eval questions (committed)
│   │   └── wiki_eval_hallucination_trap.json # hallucination-trap eval questions (committed)
│   ├── corpus_utils.py          # shared passage-sampling/JSON-parsing helpers
│   ├── generate_dataset.py      # samples corpus, generates Q&A via Groq
│   ├── generate_hallucination_dataset.py # generates hallucination-trap questions via Groq
│   ├── run_evals.py              # Ragas + OpenEvals eval runner (3 categories)
│   ├── requirements.txt          # eval-only deps
│   └── results/                  # per-run JSON reports (git-ignored)
├── tests/                       # (planned)
├── requirements.txt
└── .env.example
```

- [ ] **Step 2: Add the 3 new config rows**

Replace:

```
| `EVAL_DATASET_PATH` | `evals/dataset/wiki_eval_dataset.json` | Path to the eval Q&A dataset |
| `EVAL_SAMPLE_SIZE` | `20` | Number of Q&A pairs `generate_dataset.py` produces |
| `EVAL_RESULTS_DIR` | `evals/results` | Directory eval run reports are written to |
| `EVAL_WIKI_DATASET_DIR` | `../wiki_dataset/plain-text-wikipedia-simpleenglish` | Raw corpus dir `generate_dataset.py` samples passages from |
```

with:

```
| `EVAL_DATASET_PATH` | `evals/dataset/wiki_eval_dataset.json` | Path to the eval Q&A dataset |
| `EVAL_SAMPLE_SIZE` | `20` | Number of Q&A pairs `generate_dataset.py` produces |
| `EVAL_RESULTS_DIR` | `evals/results` | Directory eval run reports are written to |
| `EVAL_WIKI_DATASET_DIR` | `../wiki_dataset/plain-text-wikipedia-simpleenglish` | Raw corpus dir `generate_dataset.py` samples passages from |
| `EVAL_UNANSWERABLE_DATASET_PATH` | `evals/dataset/wiki_eval_unanswerable.json` | Path to the hand-curated out-of-corpus eval questions |
| `EVAL_HALLUCINATION_DATASET_PATH` | `evals/dataset/wiki_eval_hallucination_trap.json` | Path to the generated hallucination-trap eval questions |
| `EVAL_HALLUCINATION_SAMPLE_SIZE` | `10` | Number of questions `generate_hallucination_dataset.py` produces |
```

- [ ] **Step 3: Rewrite the "Evals & tests" section**

Replace the entire section (from `## Evals & tests` to the end of the file):

```
## Evals & tests

`evals/` holds retrieval/generation evals built on **Ragas** and
**OpenEvals**, using the same Groq model (`GROQ_MODEL`) as the production
pipeline as the LLM-as-judge. `tests/` is scaffolded but not yet implemented.

Install eval-only dependencies (kept separate from the prod
`requirements.txt`):

```bash
pip install -r requirements.txt -r evals/requirements.txt
```

### Generating the eval dataset

```bash
python evals/generate_dataset.py
```

Samples `EVAL_SAMPLE_SIZE` random 200-word passages from the raw corpus at
`EVAL_WIKI_DATASET_DIR` (no Chroma/Elasticsearch needed - only
`GROQ_API_KEY`), asks the Groq LLM to generate a question + reference answer
per passage (`src/prompts/eval_qa_gen.md`), and writes
`evals/dataset/wiki_eval_dataset.json`. This is committed to git as a fixed
benchmark set; re-run only when the corpus changes significantly.

### Running the evals

```bash
python evals/run_evals.py --limit 2   # fast smoke test
python evals/run_evals.py             # full dataset
```

Requires a populated Chroma collection, a reachable Elasticsearch, and
`GROQ_API_KEY` (same environment as running the FastAPI server). For each
question, runs `rag_client.answer()` and scores the result with:

- **Ragas**: `LLMContextPrecisionWithReference`, `LLMContextRecall`
  (retrieval), `Faithfulness`, `ResponseRelevancy` (generation)
- **OpenEvals**: `RAG_RETRIEVAL_RELEVANCE_PROMPT` (retrieval),
  `CORRECTNESS_PROMPT` (generation)

Prints a per-question table with a final `MEAN` row, and writes a full JSON
report to `EVAL_RESULTS_DIR/<timestamp>.json` (git-ignored). Errors for a
given question (e.g. Groq rate limits) are recorded per-question without
aborting the run - purely observational, no pass/fail gating.
```

with:

```
## Evals & tests

`evals/` holds retrieval/generation evals built on **Ragas** and
**OpenEvals**, using the same Groq model (`GROQ_MODEL`) as the production
pipeline as the LLM-as-judge. `tests/` is scaffolded but not yet implemented.

Install eval-only dependencies (kept separate from the prod
`requirements.txt`):

```bash
pip install -r requirements.txt -r evals/requirements.txt
```

### Eval categories

`run_evals.py` evaluates three categories of questions, each with its own
dataset and metric set:

- **`answerable`** (`wiki_eval_dataset.json`) - in-corpus questions with a
  known reference answer. Scored on retrieval quality
  (`LLMContextPrecisionWithReference`, `LLMContextRecall`,
  `retrieval_relevance`) and answer quality (`Faithfulness`,
  `ResponseRelevancy`, `correctness`).
- **`unanswerable`** (`wiki_eval_unanswerable.json`, hand-curated) -
  questions with no relevant source in the corpus at all. The pipeline
  should say it doesn't know rather than answer anyway.
- **`hallucination_trap`** (`wiki_eval_hallucination_trap.json`, generated) -
  questions about an in-corpus topic asking for a specific detail
  (date/number/name/award/location) that the retrieved passage does not
  state. Tests whether the model fabricates the missing detail.

`unanswerable` and `hallucination_trap` are scored on `Faithfulness`,
`retrieval_relevance`, and `hallucination` (via OpenEvals'
`HALLUCINATION_PROMPT`, which rewards appropriately indicating uncertainty
and penalizes fabricated dates/numbers/names).

### Generating the eval datasets

```bash
python evals/generate_dataset.py
python evals/generate_hallucination_dataset.py
```

Both sample random 200-word passages from the raw corpus at
`EVAL_WIKI_DATASET_DIR` (no Chroma/Elasticsearch needed - only
`GROQ_API_KEY`) and ask the Groq LLM to generate eval items:

- `generate_dataset.py` generates `EVAL_SAMPLE_SIZE` question + reference
  answer pairs (`src/prompts/eval_qa_gen.md`) and writes
  `evals/dataset/wiki_eval_dataset.json`.
- `generate_hallucination_dataset.py` generates `EVAL_HALLUCINATION_SAMPLE_SIZE`
  question + missing-detail pairs (`src/prompts/eval_hallucination_qa_gen.md`)
  and writes `evals/dataset/wiki_eval_hallucination_trap.json`.

Both outputs are committed to git as fixed benchmark sets; re-run only when
the corpus changes significantly. `wiki_eval_unanswerable.json` is a fixed,
hand-curated list - grow it by editing the file directly.

### Running the evals

```bash
python evals/run_evals.py --limit 2   # fast smoke test (2 items per category)
python evals/run_evals.py             # full dataset
```

Requires a populated Chroma collection, a reachable Elasticsearch, and
`GROQ_API_KEY` (same environment as running the FastAPI server). For each
category, runs `rag_client.answer()` per question, prints a per-question
table with a final `MEAN` row, and writes one combined JSON report to
`EVAL_RESULTS_DIR/<timestamp>.json` (git-ignored) with a `categories` key
covering all three category results and means. Errors for a given question
(e.g. Groq rate limits) are recorded per-question without aborting the run -
purely observational, no pass/fail gating.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(prod-rag): document robustness eval categories and scripts"
```
