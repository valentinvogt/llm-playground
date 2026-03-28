from pathlib import Path

from market_mvp.agents import MockLLMClient
from market_mvp.data import load_gsm8k_jsonl
from market_mvp.market import MarketEngine
from market_mvp.models import Agent


def test_runner_smoke(tmp_path: Path) -> None:
    data = load_gsm8k_jsonl("tests/fixtures/gsm8k_sample.jsonl", limit=2)
    agents = [Agent(name="a", model="m1"), Agent(name="b", model="m2"), Agent(name="c", model="m3")]
    engine = MarketEngine(agents=agents, llm_client=MockLLMClient(seed=1))

    results = [engine.run_problem(ex) for ex in data]

    assert len(results) == 2
    assert results[0].problem_id == 0
    assert all(a.budget >= a.budget_floor for a in agents)
