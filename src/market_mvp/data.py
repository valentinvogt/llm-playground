from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


@dataclass
class GSM8KExample:
    problem_id: int
    question: str
    answer_text: str
    final_answer: float


def parse_gsm8k_final_answer(answer_text: str) -> float:
    match = ANSWER_RE.search(answer_text)
    if not match:
        raise ValueError(f"Could not find final numeric answer in: {answer_text[:120]}")
    cleaned = match.group(1).replace(",", "")
    return float(cleaned)


def load_gsm8k_jsonl(path: str | Path, limit: int | None = None) -> list[GSM8KExample]:
    dataset: list[GSM8KExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            final = parse_gsm8k_final_answer(obj["answer"])
            dataset.append(
                GSM8KExample(
                    problem_id=idx,
                    question=obj["question"],
                    answer_text=obj["answer"],
                    final_answer=final,
                )
            )
            if limit is not None and len(dataset) >= limit:
                break
    return dataset
