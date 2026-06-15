# Wiki Dataset (not tracked in git)

The raw Wikipedia (Simple English) corpus used to seed Chroma/Elasticsearch
and to generate eval datasets (`prod-rag/evals/generate_dataset.py`,
`prod-rag/evals/generate_hallucination_dataset.py`) is too large for git
(the zip alone is ~130 MB, over GitHub's 100 MB file limit) and is not
committed to this repo.

## Download

Source: Kaggle dataset **"Plain Text Wikipedia (Simple English)"**
https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish

If that link has moved, search Kaggle for `plain-text-wikipedia-simpleenglish`.

## Setup

1. Download `plain-text-wikipedia-simpleenglish.zip` from the link above.
2. Extract it into this directory so the layout looks like:

   ```
   wiki_dataset/
   ├── README.md
   └── plain-text-wikipedia-simpleenglish/
       ├── 1of2/
       │   └── wiki_0, wiki_1, ... (plain-text article chunks)
       └── 2of2/
           └── wiki_0, wiki_1, ...
   ```

This matches the default `EVAL_WIKI_DATASET_DIR=../wiki_dataset/plain-text-wikipedia-simpleenglish`
in `prod-rag/.env.example`. Everything under `wiki_dataset/` except this
README is git-ignored.
