from __future__ import annotations

import random
from typing import Protocol

from .models import Agent, AgentOutcome
from .parsing import parse_steps


PROMPT_TEMPLATE = """Solve this math problem step by step.

For EACH step, output in this exact format:

STEP: <brief description>
EXPR: <mathematical expression using numbers and basic operators>
RESULT: <numerical result of evaluating EXPR>

After all steps, output:
FINAL: <final numeric answer>

Do not skip steps. Each algebraic manipulation should be its own step.

Problem:
{problem}
"""


class LLMClient(Protocol):
    def complete(self, *, model: str, prompt: str) -> str: ...


class MockLLMClient:
    """Deterministic fake client for local MVP tests without API keys."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)

    def complete(self, *, model: str, prompt: str) -> str:
        # A tiny synthetic solver for smoke testing parser/market flow.
        nums = [float(tok) for tok in prompt.replace("?", "").split() if tok.replace(".", "", 1).isdigit()]
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            jitter = self.rng.choice([0.0, 0.0, 0.0, 1.0])
            c = a + b + jitter
            return (
                "STEP: add the numbers\n"
                f"EXPR: {a} + {b}\n"
                f"RESULT: {a + b + jitter}\n"
                f"FINAL: {c}\n"
            )
        return "STEP: fallback\nEXPR: 0\nRESULT: 0\nFINAL: 0"


def run_agent(agent: Agent, question: str, client: LLMClient) -> AgentOutcome:
    raw = client.complete(model=agent.model, prompt=PROMPT_TEMPLATE.format(problem=question))
    steps, final = parse_steps(raw, agent_name=agent.name)
    parse_failed = len(steps) == 0 and final is None
    return AgentOutcome(
        agent_name=agent.name,
        steps=steps,
        final_answer=final,
        parse_failed=parse_failed,
        raw_output=raw,
    )
