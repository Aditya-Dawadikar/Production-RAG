# prod-rag Evals: Design

## Goal

Add retrieval and generation evals to `prod-rag` using **Ragas** and
**OpenEvals**, with the same LLM (`GROQ_MODEL` via the existing `ChatGroq`
instance in `src/llm_client.py`) acting as the LLM-as-judge that the
production pipeline uses for answer generation.

## Pipeline under test

```
query -> HybridRetriever (Chroma + Elasticsearch, merged/deduped)
       -> FlashRank reranker (top RERANK_TOP_K)
       -> ChatGroq (GROQ_MODEL) generation (src/prompts/rag.md)
       -> answer + sources
```

`rag_client.answer(query)` (in `src/rag.py`) returns
`{"query", "answer", "sources": [...], "metadata": {...}}` where each
source has `id`, `text`, `metadata` (`doc_id`, `source_file`, ...), and
`score`.

## File layout

```
prod-rag/
├── evals/
│   ├── dataset/
│   │   └── wiki_eval_dataset.json   # ~20 Q&A pairs, generated once, committed to git
│   ├── generate_dataset.py          # NEW - samples corpus, generates Q&A via Groq
│   ├── run_evals.py                 # eval runner (currently empty)
│   ├── requirements.txt             # NEW - eval-only deps
│   └── results/                     # NEW - gitignored, per-run JSON reports
├── src/
│   └── prompts/
│       └── eval_qa_gen.md           # NEW - prompt for dataset generation
├── .env.example                     # add EVAL_* vars
├── .gitignore (root)                # add evals/results/
└── README.md                        # document evals/ usage
```

## Dataset generation (`evals/generate_dataset.py`)

One-time / occasional script, run manually when the corpus changes
significantly. Output is committed to git so `run_evals.py` always
evaluates against a fixed, comparable benchmark set.

Source data is the raw corpus at `wiki_dataset/plain-text-wikipedia-simpleenglish/`
(repo root, sibling of `prod-rag/`, tracked in git) — the same input
`spark-preprocessing/src/preprocess_wiki.py` chunks for the production
index. Using this directly means `generate_dataset.py` needs **no live
Chroma/ES connection** — only `GROQ_API_KEY`.

1. List all files under `1of2/` and `2of2/` (171 files total:
   `wiki_00`...`wiki_99` in `1of2`, `wiki_00`...`wiki_70` in `2of2`).
2. Pick `EVAL_SAMPLE_SIZE` (default `20`) distinct random files. For each:
   - Read the file and normalize whitespace the same way as
     `preprocess_wiki.clean_text` (`re.sub(r"\s+", " ", text).strip()`),
     producing one long word sequence per file (mirrors the
     `wholetext=True` read in the Spark job).
   - Pick a random 200-word window (matching the production
     `chunk_size=200` default) as the source passage.
3. For each sampled passage, prompt the Groq LLM (`GROQ_MODEL`, via
   `src.llm_client.llm_client.llm`) using a new prompt template
   `src/prompts/eval_qa_gen.md`: given the passage, produce one factual
   question answerable from it and a concise reference answer.
4. Write `evals/dataset/wiki_eval_dataset.json` as a JSON list of:
   ```json
   {
     "question": "...",
     "reference": "...",
     "source_file": "1of2/wiki_07"
   }
   ```
   `source_file` is kept only for debugging/traceability (which raw file the
   passage came from) — it does not need to match any `doc_id`/`chunk_id`
   in the live index.
5. Skip/retry passages where the LLM fails to produce a usable
   question/answer (e.g. malformed output), so the script still converges
   on `EVAL_SAMPLE_SIZE` usable pairs.

## Eval runner (`evals/run_evals.py`)

1. Load the dataset from `EVAL_DATASET_PATH` (default
   `evals/dataset/wiki_eval_dataset.json`). Optional `--limit N` CLI flag
   runs on the first N items only, for fast iteration.
2. For each `{question, reference, source_file}`:
   - Call `rag_client.answer(question)` to get `answer` and `sources`.
   - Build a Ragas `SingleTurnSample`:
     - `user_input = question`
     - `response = answer`
     - `retrieved_contexts = [s["text"] for s in sources]`
     - `reference = reference`
3. **Judges — same model as production:**
   - Ragas LLM: `LangchainLLMWrapper(llm_client.llm)` (the existing
     `ChatGroq` instance, so judge model == `GROQ_MODEL` with the same
     temperature/config as generation).
   - Ragas embeddings: `LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))`
     (same embedding model used for retrieval).
   - OpenEvals judges: `create_llm_as_judge(model=f"groq:{GROQ_MODEL}", ...)`.
4. **Ragas metrics** via `ragas.evaluate(dataset, metrics=[...], llm=..., embeddings=...)`:
   - Retrieval: `LLMContextPrecisionWithReference`, `LLMContextRecall`
   - Generation: `Faithfulness`, `ResponseRelevancy`
5. **OpenEvals metrics**, run per-question via `create_llm_as_judge`:
   - Retrieval: `RAG_RETRIEVAL_RELEVANCE_PROMPT` — are the retrieved
     contexts relevant to the question?
   - Generation: `CORRECTNESS_PROMPT` — is the answer correct relative to
     `reference`?
6. **Error isolation**: if `rag_client.answer()` or any metric computation
   raises for a given question (e.g. Groq rate limit/timeout), record an
   error entry for that question in the report and continue with the rest
   of the dataset rather than aborting the run.
7. **Output**:
   - Console: a per-question table (truncated question text + all metric
     scores), plus a final `MEAN` row with per-metric averages.
   - JSON report written to `evals/results/<timestamp>.json` containing:
     full per-question detail (question, answer, retrieved contexts,
     reference, all scores, any error), aggregate means, and run metadata
     (`GROQ_MODEL`, `EMBEDDING_MODEL_NAME`, dataset path, timestamp,
     retrieval/rerank top-k).
   - No pass/fail gating — purely observational for now.

## Config additions

Add to `.env.example` and the README config table:

| Variable | Default | Description |
| --- | --- | --- |
| `EVAL_DATASET_PATH` | `evals/dataset/wiki_eval_dataset.json` | Path to the eval Q&A dataset |
| `EVAL_SAMPLE_SIZE` | `20` | Number of Q&A pairs `generate_dataset.py` produces |
| `EVAL_RESULTS_DIR` | `evals/results` | Directory eval run reports are written to |
| `EVAL_WIKI_DATASET_DIR` | `../wiki_dataset/plain-text-wikipedia-simpleenglish` | Raw corpus dir `generate_dataset.py` samples passages from |

All other config (`GROQ_API_KEY`, `GROQ_MODEL`, `EMBEDDING_MODEL_NAME`,
`RETRIEVAL_TOP_K`, `RERANK_TOP_K`, Chroma/ES settings) is reused as-is from
the existing `.env`. `generate_dataset.py` imports the already-configured
`llm_client`; `run_evals.py` imports `rag_client` (which itself wires up
`llm_client`, `chromadb_client`, and `elasticsearch_client`).

## Dependencies

New `prod-rag/evals/requirements.txt` (kept separate from the prod
`requirements.txt` so the deployed server's footprint is unchanged):

- `ragas`
- `openevals`
- `langchain-huggingface`

Install for running evals:
```
pip install -r requirements.txt -r evals/requirements.txt
```

## .gitignore

Add `evals/results/` (per-run JSON reports are not committed).

## Testing / validation

- `generate_dataset.py` only needs a valid `GROQ_API_KEY` and read access
  to `wiki_dataset/plain-text-wikipedia-simpleenglish/` (repo root) — no
  Chroma/ES required. Validate by running once and inspecting
  `wiki_eval_dataset.json` for well-formed, non-empty
  `question`/`reference`/`source_file` fields across all
  `EVAL_SAMPLE_SIZE` entries.
- `run_evals.py` requires a populated Chroma collection, a reachable
  Elasticsearch (since `rag_client.answer()` uses hybrid retrieval), and a
  valid `GROQ_API_KEY` — the same environment as running the FastAPI server
  locally or on the EC2 box. Validate with `--limit 2` first (fast smoke
  test of the Ragas + OpenEvals wiring and console/JSON output), then run on
  the full dataset.

## Out of scope / future work

- No pytest integration or CI gating (purely observational reports for now).
- No pass/fail thresholds.
- `tests/` directory remains untouched — this work is scoped to `evals/`.
- Re-running `generate_dataset.py` to refresh the benchmark set is a manual,
  occasional operation, not automated.
