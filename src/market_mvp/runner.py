from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import MockLLMClient
from .data import load_gsm8k_jsonl
from .market import MarketEngine
from .metrics import summarize, write_budget_history, write_problem_log, write_settlement_log
from .models import Agent
from .plots import plot_budget_evolution, plot_calibration, plot_hard_problem_accuracy


def build_default_agents() -> list[Agent]:
    return [
        Agent(name="agent_haiku", model="claude-haiku"),
        Agent(name="agent_sonnet", model="claude-sonnet"),
        Agent(name="agent_gpt4omini", model="gpt-4o-mini"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run market-based LLM reasoning MVP.")
    parser.add_argument("--gsm8k-path", required=True, help="Path to GSM8K test jsonl.")
    parser.add_argument("--limit", type=int, default=100, help="Number of problems to run.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for logs and plots.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_gsm8k_jsonl(args.gsm8k_path, limit=args.limit)
    client = MockLLMClient()
    agents = build_default_agents()
    engine = MarketEngine(agents=agents, llm_client=client)

    results = [engine.run_problem(ex) for ex in examples]

    write_problem_log(results, output_dir / "problem_results.csv")
    write_settlement_log(results, output_dir / "settlements.json")
    write_budget_history(agents, output_dir / "budget_history.json")

    summary = summarize(results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_calibration(results, output_dir / "calibration_curve.png")
    plot_budget_evolution(agents, output_dir / "budget_evolution.png")
    plot_hard_problem_accuracy(results, output_dir / "hard_problem_accuracy.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
