from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    """A single structured reasoning step proposed by an agent."""

    description: str
    expression: str
    result: float
    agent_name: str


@dataclass
class AgentOutcome:
    """All generated artifacts and outcomes for one agent on one problem."""

    agent_name: str
    steps: list[Step] = field(default_factory=list)
    final_answer: float | None = None
    parse_failed: bool = False
    raw_output: str | None = None


@dataclass
class StepSettlement:
    """Settlement details for one agent at one aligned step index."""

    problem_id: int
    step_index: int
    agent_name: str
    bet: float
    verified: bool | None
    delta: float
    budget_after: float


@dataclass
class ProblemResult:
    """Aggregate result payload for one problem across all methods."""

    problem_id: int
    question: str
    ground_truth: float
    market_final_answer: float | None
    market_correct: bool
    single_best_answer: float | None
    single_best_correct: bool
    majority_answer: float | None
    majority_correct: bool
    best_of_n_answer: float | None
    best_of_n_correct: bool
    per_agent: dict[str, AgentOutcome]
    settlements: list[StepSettlement]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """Mutable market agent state and behavior."""

    name: str
    model: str
    budget: float = 1000.0
    budget_floor: float = 100.0
    budget_history: list[float] = field(default_factory=list)
    step_record: list[dict[str, Any]] = field(default_factory=list)

    def clamp_budget(self) -> None:
        if self.budget < self.budget_floor:
            self.budget = self.budget_floor
        self.budget_history.append(self.budget)
