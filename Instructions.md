# Market-Based LLM Reasoning Aggregation — MVP Experiment

## Concept

Multiple LLM agents solve multi-step math problems. Instead of majority voting or best-of-N, agents **bet virtual currency** on each reasoning step. Steps are mechanically verified. Agents that bet on correct steps gain budget; agents that bet on incorrect steps lose budget. Over many problems, budget dynamics should produce better calibration (step-level confidence) and potentially better accuracy on hard problems than baselines.

## Hypothesis

1. Market spread (agreement/disagreement of bets) predicts step-level correctness better than any individual model's logit confidence.
2. On hard problems (where individual models fail >30% of the time), the market outperforms majority vote on accuracy by 5–10 points because it reallocates influence mid-problem based on per-step verification.
3. Agent budgets after many problems reveal per-topic capability profiles (e.g., model X is good at fractions, bad at geometry).

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Experiment Runner               │
│  Loops over problems, collects metrics, logs     │
└──────────────┬──────────────────────┬────────────┘
               │                      │
       ┌───────▼───────┐     ┌────────▼────────┐
       │  Market Engine │     │   Baselines     │
       │  (core logic)  │     │  (majority vote,│
       │                │     │   best-of-N)    │
       └───┬───┬───┬────┘     └─────────────────┘
           │   │   │
     ┌─────▼┐ ┌▼────┐ ┌▼─────┐
     │AgentA│ │AgentB│ │AgentC│  ... (N agents, each wrapping an LLM)
     └──────┘ └──────┘ └──────┘
                  │
          ┌───────▼────────┐
          │   Verifier     │
          │ (sympy-based)  │
          └────────────────┘
```

## Dataset

Use **GSM8K** (grade school math). It's standard, well-studied, and has ground-truth final answers.

Download: https://github.com/openai/grade-school-math (use the `test` split, 1319 problems).

Each problem is a word problem with a numeric final answer. Example:

> "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
> Answer: 72

## Step-by-Step Implementation

### 1. Data Loading

Parse GSM8K jsonl. Each entry has `question` and `answer` fields. The answer field contains chain-of-thought followed by `#### <numeric_answer>`. Extract the numeric final answer for ground-truth comparison.

### 2. Agent Interface

Each agent wraps an LLM API call. Use the Anthropic API (or OpenAI, or a mix — using different models is the point).

```python
class Agent:
    def __init__(self, name: str, model: str, budget: float = 1000.0):
        self.name = name
        self.model = model
        self.budget = budget
        self.budget_history = []  # track over time
        self.step_record = []     # per-step bets and outcomes

    def propose_solution(self, problem: str) -> list[Step]:
        """
        Prompt the LLM to solve the problem step-by-step.
        Return a list of Step objects.
        """
        ...

    def bet_on_step(self, step: Step, competing_steps: list[Step]) -> float:
        """
        Given a proposed step (possibly from another agent), return
        a bet amount (0 to self.budget). Higher = more confident this
        step is correct.
        """
        ...
```

Each `Step` is:

```python
@dataclass
class Step:
    description: str          # natural language explanation
    expression: str           # the mathematical expression, e.g. "x = 500 / 3.6"
    result: float             # numerical result of this step
    agent_name: str           # who proposed it
```

### 3. Prompting Strategy

Use a structured prompt that forces step-by-step output in a parseable format. Example system prompt:

```
Solve this math problem step by step.

For EACH step, output in this exact format:

STEP: <brief description>
EXPR: <mathematical expression using numbers and basic operators>
RESULT: <numerical result of evaluating EXPR>

After all steps, output:
FINAL: <final numeric answer>

Do not skip steps. Each algebraic manipulation should be its own step.
```

Parse the LLM output into `Step` objects. If parsing fails, the agent forfeits that problem (loses a fixed penalty from budget).

### 4. Verifier

Use `sympy` to verify each step mechanically.

```python
def verify_step(expression: str, claimed_result: float, tolerance: float = 1e-6) -> bool:
    """
    Evaluate `expression` symbolically and check if it equals `claimed_result`.
    """
    try:
        actual = float(sympy.sympify(expression))
        return abs(actual - claimed_result) < tolerance
    except:
        return False
```

Also verify the final answer against GSM8K ground truth.

### 5. Market Mechanism

For each problem:

```
1. All N agents independently produce a full step-by-step solution.
2. Align steps across agents (by step index — step 1 vs step 1, etc).
   If agents have different numbers of steps, pad shorter solutions.
3. For each step position:
   a. Each agent's proposed step is verified independently.
   b. Each agent bets on its own step. Bet = self-assessed confidence.
      For the MVP, use a simple heuristic: agents bet a fixed fraction
      of their remaining budget (e.g., budget * 0.05 per step).
      STRETCH GOAL: prompt the LLM to self-assess confidence as a
      number 1-10, and scale the bet accordingly.
   c. Settlement:
      - If the step is VERIFIED CORRECT: agent keeps its bet and gains
        a share of the losing agents' bets on that step position,
        proportional to bet size.
      - If the step is VERIFIED INCORRECT: agent loses its bet.
        The lost amount is distributed to correct agents.
      - If verification is INCONCLUSIVE (expression unparseable):
        bets are returned (no settlement).
4. After all steps: check final answer against ground truth.
   Agents with correct final answers get a bonus (e.g., 50 units).
   Agents with incorrect final answers pay a penalty (e.g., 30 units).
5. Record all budget changes, bets, and verification outcomes.
```

### 6. Baselines to Compare Against

Implement these using the **same models and same number of API calls**:

- **Single best model**: just call the strongest model once.
- **Majority vote**: all N agents solve independently, take the most common final answer.
- **Best-of-N with verifier**: all N agents solve independently, pick the first one where all steps verify and the final answer is consistent. If none fully verify, fall back to majority vote.

### 7. Metrics to Collect

Per-problem:
- Final answer correctness (binary) for market vs each baseline.
- Per-step verification results.
- Market spread at each step: standard deviation of bets across agents. This is the main calibration signal.

Aggregate:
- Overall accuracy (% problems correct) for market vs baselines.
- **Calibration plot**: bin steps by market spread, plot fraction of steps that were actually correct per bin. A well-calibrated market means tight spread → high correctness, wide spread → low correctness.
- **Hard-problem accuracy**: filter to problems where the single best model gets it wrong, compare market vs baselines on this subset.
- **Budget evolution**: plot each agent's budget over time. Look for specialization (does any agent's budget correlate with problem subtypes?).
- **Comparison to logit entropy**: if using an API that returns logprobs, compare market spread vs token-level entropy as a predictor of step correctness (ROC curves).

### 8. Suggested Model Selection

Use 3-5 models of **varying capability** to make the market dynamics interesting. For example:
- Claude Haiku (fast, cheap, makes more errors — interesting market participant)
- Claude Sonnet (mid-tier)
- GPT-4o-mini
- Gemini Flash
- One open-source model via a local endpoint if available

Heterogeneity matters. 5 copies of the same model will converge to the same answers and the market adds nothing.

### 9. Implementation Order

1. Data loader for GSM8K.
2. Step parser (structured output → Step objects).
3. Sympy verifier.
4. Single-agent pipeline (one model, end-to-end, verify steps, check answer).
5. Multi-agent with majority vote baseline.
6. Market mechanism on top of multi-agent.
7. Metrics collection and plotting.
8. Run on 100 problems first (fast iteration), then full 1319.

### 10. Output

Produce:
- A CSV/JSON log of every problem: agents' steps, bets, verification results, budget changes, final answers.
- Summary statistics table: accuracy and calibration for market vs each baseline.
- Plots: calibration curve, budget evolution over time, hard-problem accuracy comparison.

### 11. Key Implementation Pitfalls to Avoid

- **Don't over-engineer the market.** A simple proportional bet + winner-takes-losers'-pool is enough for the MVP. No need for order books or continuous auctions.
- **Step alignment across agents is tricky.** Agents may decompose problems differently (3 steps vs 5 steps). For the MVP, just compare final answers if step counts don't match. Only run the step-level market when agents produce the same number of steps.
- **Budget floors.** Don't let agents go to zero budget — set a minimum (e.g., 100) so they can always participate. Otherwise one bad problem eliminates an agent permanently.
- **Parsing failures will be common.** LLMs don't reliably produce structured math output. Build robust parsing with fallbacks. Log parse failure rates per model — this is itself interesting data.
- **Rate limits.** With 5 models × 1319 problems, you're making ~6500+ API calls minimum. Add delays and retry logic. Budget 2-4 hours of API time and maybe $20-50 in API costs depending on models.

### 12. Stretch Goals (not MVP)

- Let agents bet on *other* agents' steps, not just their own.
- Prompt agents to see other agents' proposals before betting (creates an actual information market rather than independent assessment).
- Track per-problem-category budgets (problems involving fractions, percentages, geometry, etc.) to quantify emergent specialization.
- Compare against a reward-model-based ranker as an additional baseline.
