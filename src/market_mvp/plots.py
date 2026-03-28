from __future__ import annotations

import csv
from pathlib import Path

from .models import Agent, ProblemResult

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _fallback_write_table(path: str | Path, rows: list[dict[str, float]]) -> None:
    p = Path(path)
    p = p.with_suffix(".csv")
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["x", "y"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_budget_evolution(agents: list[Agent], path: str | Path) -> None:
    if plt is None:
        rows = []
        for agent in agents:
            for idx, budget in enumerate(agent.budget_history):
                rows.append({"agent": agent.name, "idx": idx, "budget": budget})
        _fallback_write_table(path, rows)
        return

    plt.figure(figsize=(8, 4))
    for agent in agents:
        plt.plot(agent.budget_history, label=agent.name)
    plt.title("Agent Budget Evolution")
    plt.xlabel("Settlement event")
    plt.ylabel("Budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_hard_problem_accuracy(results: list[ProblemResult], path: str | Path) -> None:
    hard = [r for r in results if not r.single_best_correct] or results
    vals = {
        "Market": sum(r.market_correct for r in hard) / max(1, len(hard)),
        "Majority": sum(r.majority_correct for r in hard) / max(1, len(hard)),
        "Best-of-N": sum(r.best_of_n_correct for r in hard) / max(1, len(hard)),
    }

    if plt is None:
        _fallback_write_table(path, [{"method": k, "accuracy": v} for k, v in vals.items()])
        return

    plt.figure(figsize=(6, 4))
    plt.bar(list(vals.keys()), list(vals.values()))
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Hard-problem Accuracy")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_calibration(results: list[ProblemResult], path: str | Path) -> None:
    spreads: list[float] = []
    correctness: list[int] = []
    for r in results:
        if r.settlements:
            spread = sum(abs(s.bet) for s in r.settlements) / len(r.settlements)
            spreads.append(spread)
            correctness.append(1 if r.market_correct else 0)
    if not spreads:
        spreads = [0.0]
        correctness = [0]

    min_s, max_s = min(spreads), max(spreads)
    bin_count = 5
    width = (max_s - min_s) / bin_count if max_s > min_s else 1.0

    points: list[dict[str, float]] = []
    for i in range(bin_count):
        lo = min_s + i * width
        hi = lo + width
        idxs = [j for j, s in enumerate(spreads) if lo <= s < hi or (i == bin_count - 1 and s == hi)]
        if not idxs:
            continue
        points.append({"spread_center": (lo + hi) / 2, "fraction_correct": sum(correctness[j] for j in idxs) / len(idxs)})

    if plt is None:
        _fallback_write_table(path, points)
        return

    plt.figure(figsize=(6, 4))
    plt.plot([p["spread_center"] for p in points], [p["fraction_correct"] for p in points], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Market spread (proxy)")
    plt.ylabel("Fraction correct")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
