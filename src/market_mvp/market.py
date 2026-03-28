from __future__ import annotations

import statistics
from collections import Counter

from .agents import LLMClient, run_agent
from .data import GSM8KExample
from .models import Agent, AgentOutcome, ProblemResult, StepSettlement
from .parsing import aligned_step_count
from .verifier import verify_final, verify_step


class MarketEngine:
    def __init__(
        self,
        agents: list[Agent],
        llm_client: LLMClient,
        bet_fraction: float = 0.05,
        parse_failure_penalty: float = 25.0,
        final_correct_bonus: float = 50.0,
        final_wrong_penalty: float = 30.0,
    ) -> None:
        self.agents = agents
        self.client = llm_client
        self.bet_fraction = bet_fraction
        self.parse_failure_penalty = parse_failure_penalty
        self.final_correct_bonus = final_correct_bonus
        self.final_wrong_penalty = final_wrong_penalty

    def run_problem(self, ex: GSM8KExample) -> ProblemResult:
        outcomes = {a.name: run_agent(a, ex.question, self.client) for a in self.agents}

        for agent in self.agents:
            if outcomes[agent.name].parse_failed:
                agent.budget -= self.parse_failure_penalty
                agent.clamp_budget()

        settlements: list[StepSettlement] = []
        aligned = aligned_step_count([o.steps for o in outcomes.values() if not o.parse_failed])

        if aligned is not None:
            for step_idx in range(aligned):
                bets: dict[str, float] = {}
                verdicts: dict[str, bool | None] = {}
                for agent in self.agents:
                    step = outcomes[agent.name].steps[step_idx]
                    bet = max(0.0, min(agent.budget * self.bet_fraction, agent.budget - agent.budget_floor))
                    bets[agent.name] = bet
                    verdicts[agent.name] = verify_step(step.expression, step.result)

                inconclusive = all(v is None for v in verdicts.values())
                if inconclusive:
                    deltas = {name: 0.0 for name in bets}
                else:
                    winners = [n for n, v in verdicts.items() if v is True]
                    losers = [n for n, v in verdicts.items() if v is False]
                    pool = sum(bets[n] for n in losers)
                    deltas = {name: 0.0 for name in bets}

                    for loser in losers:
                        deltas[loser] -= bets[loser]

                    if winners:
                        winner_bet_total = sum(bets[n] for n in winners)
                        if winner_bet_total > 0:
                            for winner in winners:
                                deltas[winner] += pool * (bets[winner] / winner_bet_total)

                for agent in self.agents:
                    delta = deltas[agent.name]
                    agent.budget += delta
                    agent.clamp_budget()
                    settlements.append(
                        StepSettlement(
                            problem_id=ex.problem_id,
                            step_index=step_idx,
                            agent_name=agent.name,
                            bet=bets[agent.name],
                            verified=verdicts[agent.name],
                            delta=delta,
                            budget_after=agent.budget,
                        )
                    )

        for agent in self.agents:
            is_correct = verify_final(outcomes[agent.name].final_answer, ex.final_answer)
            agent.budget += self.final_correct_bonus if is_correct else -self.final_wrong_penalty
            agent.clamp_budget()

        market_ans = self._market_final_answer(outcomes)
        single_best = self._single_best_baseline(outcomes)
        majority = self._majority_vote_baseline(outcomes)
        best_of_n = self._best_of_n_with_verifier(outcomes, ex.final_answer)

        spread_values = [s.bet for s in settlements]

        return ProblemResult(
            problem_id=ex.problem_id,
            question=ex.question,
            ground_truth=ex.final_answer,
            market_final_answer=market_ans,
            market_correct=verify_final(market_ans, ex.final_answer),
            single_best_answer=single_best,
            single_best_correct=verify_final(single_best, ex.final_answer),
            majority_answer=majority,
            majority_correct=verify_final(majority, ex.final_answer),
            best_of_n_answer=best_of_n,
            best_of_n_correct=verify_final(best_of_n, ex.final_answer),
            per_agent=outcomes,
            settlements=settlements,
            meta={
                "aligned_steps": aligned,
                "mean_bet": statistics.mean(spread_values) if spread_values else 0.0,
            },
        )

    def _single_best_baseline(self, outcomes: dict[str, AgentOutcome]) -> float | None:
        return outcomes[self.agents[0].name].final_answer

    def _majority_vote_baseline(self, outcomes: dict[str, AgentOutcome]) -> float | None:
        answers = [o.final_answer for o in outcomes.values() if o.final_answer is not None]
        if not answers:
            return None
        return Counter(answers).most_common(1)[0][0]

    def _best_of_n_with_verifier(self, outcomes: dict[str, AgentOutcome], ground_truth: float) -> float | None:
        for agent in self.agents:
            outcome = outcomes[agent.name]
            if not outcome.steps:
                continue
            if all(verify_step(s.expression, s.result) is True for s in outcome.steps):
                if verify_final(outcome.final_answer, ground_truth):
                    return outcome.final_answer
        return self._majority_vote_baseline(outcomes)

    def _market_final_answer(self, outcomes: dict[str, AgentOutcome]) -> float | None:
        weighted: dict[float, float] = {}
        for agent in self.agents:
            ans = outcomes[agent.name].final_answer
            if ans is None:
                continue
            weighted[ans] = weighted.get(ans, 0.0) + agent.budget
        if not weighted:
            return None
        return max(weighted.items(), key=lambda x: x[1])[0]
