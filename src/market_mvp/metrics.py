from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import Agent, ProblemResult


def summarize(results: list[ProblemResult]) -> dict[str, float]:
    total = max(1, len(results))
    return {
        "n_problems": len(results),
        "market_accuracy": sum(r.market_correct for r in results) / total,
        "single_best_accuracy": sum(r.single_best_correct for r in results) / total,
        "majority_accuracy": sum(r.majority_correct for r in results) / total,
        "best_of_n_accuracy": sum(r.best_of_n_correct for r in results) / total,
    }


def write_problem_log(results: list[ProblemResult], path: str | Path) -> None:
    rows = []
    for r in results:
        rows.append(
            {
                "problem_id": r.problem_id,
                "ground_truth": r.ground_truth,
                "market_answer": r.market_final_answer,
                "market_correct": r.market_correct,
                "single_best_answer": r.single_best_answer,
                "single_best_correct": r.single_best_correct,
                "majority_answer": r.majority_answer,
                "majority_correct": r.majority_correct,
                "best_of_n_answer": r.best_of_n_answer,
                "best_of_n_correct": r.best_of_n_correct,
                "aligned_steps": r.meta.get("aligned_steps"),
                "mean_bet": r.meta.get("mean_bet"),
            }
        )

    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["problem_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_settlement_log(results: list[ProblemResult], path: str | Path) -> None:
    data = []
    for r in results:
        for s in r.settlements:
            data.append(
                {
                    "problem_id": s.problem_id,
                    "step_index": s.step_index,
                    "agent_name": s.agent_name,
                    "bet": s.bet,
                    "verified": s.verified,
                    "delta": s.delta,
                    "budget_after": s.budget_after,
                }
            )
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_budget_history(agents: list[Agent], path: str | Path) -> None:
    payload = {a.name: a.budget_history for a in agents}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
