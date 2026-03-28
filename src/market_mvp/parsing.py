from __future__ import annotations

import re
from typing import Iterable

from .models import Step


STEP_BLOCK_RE = re.compile(
    r"STEP:\s*(?P<description>.*?)\nEXPR:\s*(?P<expr>.*?)\nRESULT:\s*(?P<result>.*?)(?=\nSTEP:|\nFINAL:|\Z)",
    flags=re.DOTALL,
)
FINAL_RE = re.compile(r"FINAL:\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def parse_steps(raw: str, agent_name: str) -> tuple[list[Step], float | None]:
    steps: list[Step] = []
    for match in STEP_BLOCK_RE.finditer(raw.strip()):
        desc = match.group("description").strip()
        expr = match.group("expr").strip()
        result_text = match.group("result").strip().replace(",", "")
        try:
            result = float(result_text)
        except ValueError:
            continue
        steps.append(Step(description=desc, expression=expr, result=result, agent_name=agent_name))

    final_answer: float | None = None
    final_match = FINAL_RE.search(raw)
    if final_match:
        final_answer = float(final_match.group(1).replace(",", ""))
    return steps, final_answer


def aligned_step_count(step_lists: Iterable[list[Step]]) -> int | None:
    lengths = {len(s) for s in step_lists}
    if not lengths:
        return None
    return next(iter(lengths)) if len(lengths) == 1 else None
