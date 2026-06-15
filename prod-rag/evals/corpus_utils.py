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
