# Market-Based LLM Reasoning Aggregation (MVP)

This repository contains an MVP implementation of the experiment described in `Instructions.md`.

## What is implemented

- GSM8K JSONL loader with `#### <answer>` extraction.
- Structured step parser for:
  - `STEP:`
  - `EXPR:`
  - `RESULT:`
  - `FINAL:`
- Mechanical verification for steps (uses Sympy when available, with a safe arithmetic fallback).
- Multi-agent market engine with:
  - fixed-fraction per-step betting,
  - winner-takes-losers settlement,
  - parse failure penalties,
  - final-answer bonus/penalty,
  - budget floor enforcement.
- Baselines:
  - single best model (first agent),
  - majority vote,
  - best-of-N with verifier fallback to majority.
- Output artifacts:
  - `problem_results.csv`,
  - `settlements.json`,
  - `budget_history.json`,
  - `summary.json`,
  - calibration/budget/hard-problem plots.

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m market_mvp.runner --gsm8k-path /path/to/gsm8k_test.jsonl --limit 100 --output-dir outputs
```

## Notes

- The default runner uses a `MockLLMClient` so the full pipeline can run without API keys.
- Replace `MockLLMClient` with a real API-backed implementation of `LLMClient.complete(...)` for live experiments.
- If agents emit different step counts, the market skips step-level settlement for that problem and still applies final-answer scoring.
