"""Core routing logic.

Given a natural-language description of a task, ask Claude (Opus 4.8) to
recommend which of three models to actually run the task on:

    - Claude Sonnet 4.6  ("sonnet") — fast, cost-effective, high-volume work
    - Claude Opus 4.8    ("opus")   — most capable Opus; hard reasoning + agentic
    - Claude Fable 5     ("fable")  — most capable overall; the hardest tasks

The router itself runs on Opus 4.8 and returns a structured recommendation.
"""

from __future__ import annotations

from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field

# The model that does the routing (the "judge").
ROUTER_MODEL = "claude-opus-4-8"

# The candidate models the router chooses between, mapped to their API IDs.
CANDIDATES: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    "fable": "claude-fable-5",
}

ModelChoice = Literal["sonnet", "opus", "fable"]

SYSTEM_PROMPT = """\
You are a model-selection router. Given a description of a task a user wants \
to run on a large language model, decide which of exactly three Claude models \
is the best fit, balancing capability, latency, and cost.

The three candidates:

- "sonnet" (Claude Sonnet 4.6): the best balance of speed and intelligence.
  Input $3 / output $15 per million tokens. Choose it for high-volume,
  latency- or cost-sensitive work that is well-specified and not especially
  hard: summarization, classification, extraction, routine chat, drafting,
  straightforward code, and standard tool-use pipelines.

- "opus" (Claude Opus 4.8): the most capable Opus-tier model. Input $5 /
  output $25 per million tokens. Choose it for genuinely hard reasoning,
  long-horizon agentic work, complex multi-file coding, careful analysis, and
  knowledge work where quality clearly matters more than cost.

- "fable" (Claude Fable 5): the most capable model overall, for the most
  demanding reasoning and long-horizon autonomous work. Input $10 / output
  $50 per million tokens — the most expensive. Reserve it for the hardest,
  highest-stakes tasks where Opus would plausibly fall short: frontier
  research-grade reasoning, very long autonomous runs, or work where a wrong
  answer is very costly. Do NOT pick Fable for routine work just because it is
  the strongest — its cost should be justified by task difficulty or stakes.

Decision guidance:
- Default to the cheapest model that comfortably handles the task. Escalate
  only when complexity, stakes, or required autonomy justify it.
- Weigh: intrinsic difficulty, how well-specified the task is, the cost of an
  error, latency sensitivity, cost sensitivity, and how long/autonomous the
  run is.
- If the user states constraints (e.g. "this is high-volume", "cost is
  critical", "this is mission-critical"), honor them.

Always provide a clear recommendation, a runner-up, an honest confidence, and
concise reasoning."""


class ModelRecommendation(BaseModel):
    """Structured recommendation returned by the router."""

    recommended_model: ModelChoice = Field(
        description="The single best-fit model for this task."
    )
    runner_up: Optional[ModelChoice] = Field(
        default=None,
        description="The next-best alternative, if any.",
    )
    complexity: Literal["low", "medium", "high"] = Field(
        description="Assessed intrinsic difficulty of the task."
    )
    confidence: float = Field(
        description="Confidence in the recommendation, from 0.0 to 1.0."
    )
    reasoning: str = Field(
        description="Concise explanation of why this model was chosen."
    )
    considerations: list[str] = Field(
        description="Key factors that drove the decision (cost, latency, "
        "difficulty, stakes, etc.)."
    )

    @property
    def model_id(self) -> str:
        """The concrete API model ID for the recommended model."""
        return CANDIDATES[self.recommended_model]


def recommend_model(
    task: str,
    *,
    client: anthropic.Anthropic | None = None,
) -> ModelRecommendation:
    """Recommend which model to use for ``task``.

    Args:
        task: A natural-language description of the task to be performed.
        client: An optional pre-configured Anthropic client. If omitted, a
            default client is created (reads ``ANTHROPIC_API_KEY`` from the
            environment).

    Returns:
        A :class:`ModelRecommendation`.

    Raises:
        ValueError: If ``task`` is empty, or the router refuses / returns no
            parseable result.
    """
    task = task.strip()
    if not task:
        raise ValueError("task description is empty")

    client = client or anthropic.Anthropic()

    response = client.messages.parse(
        model=ROUTER_MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Recommend a model for this task:\n\n{task}",
            }
        ],
        output_format=ModelRecommendation,
    )

    if response.stop_reason == "refusal":
        raise ValueError("the router declined to answer this request")

    recommendation = response.parsed_output
    if recommendation is None:
        raise ValueError("the router returned no parseable recommendation")

    return recommendation
