"""Typed client for TypeSafe's Jev via OpenRouter's Decisions API.

Jev does not generate text. You send a `state` (any JSON) plus typed questions
about it, and get back calibrated probabilities. There are exactly three
question types: noul (yes/no), choice (pick one), score (ordered scale).
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal, Mapping, Sequence, Union

import requests
from pydantic import BaseModel, Field

from .cassette import Cassette

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "~typesafe/jev-latest"


# --------------------------------------------------------------------------- questions
class NoulQuestion(BaseModel):
    """Yes/no. The answer is the probability that the statement is true."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[str, str] | None = None  # {"true": ..., "false": ...}


class ChoiceQuestion(BaseModel):
    """Pick one of `criteria`. The answer carries a probability per option."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]  # {option_key: what it means}


class ScoreQuestion(BaseModel):
    """Position on an ordered scale. `criteria` are the levels, lowest first."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


Question = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


# --------------------------------------------------------------------------- answers
class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float  # probability of "yes", 0..1

    @property
    def yes(self) -> bool:
        return self.noul > 0.5

    @property
    def uncertain(self) -> bool:
        return 0.3 < self.noul < 0.7


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float = 0.0  # how concentrated the distribution is, NOT p(correct)


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float  # expected position, e.g. 1.05 ≈ level index 1
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0

    @property
    def level(self) -> str:
        return self.legend[str(round(self.score))]


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class Decision(BaseModel):
    model: str = ""
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    id: str | None = None
    provider: str | None = None


class JevAPIError(RuntimeError):
    """Non-200 from the API. `body` holds the parsed error payload."""

    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        super().__init__(f"Jev API error {status}: {json.dumps(body)[:400]}")

    @property
    def message(self) -> str:
        err = self.body.get("error") if isinstance(self.body, dict) else None
        if isinstance(err, dict):
            return str(err.get("message", ""))
        return str(err or self.body)


# --------------------------------------------------------------------------- client
class JevClient:
    """Calls Jev, optionally through a cassette so tests can run offline.

    Modes (env `JEV_MODE`, or the `cassette` argument):
      replay  use recorded responses only, no network       (default)
      record  call the API and append every response to the cassette
      live    call the API, record nothing
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        cassette: Cassette | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.cassette = cassette
        self.timeout = timeout
        self.total_cost = 0.0
        self.calls = 0

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Question | Mapping[str, Any]],
        *,
        model: str | None = None,
    ) -> Decision:
        """Ask one or more questions about `state`.

        Questions in a single request are evaluated in parallel, so asking ten
        costs barely more time than asking one.
        """
        payload = {
            "model": model or self.model,
            "state": state,
            "questions": {
                key: (q.model_dump(exclude_none=True) if isinstance(q, BaseModel) else dict(q))
                for key, q in questions.items()
            },
        }
        status, body = self._send(payload)
        if status != 200:
            raise JevAPIError(status, body)
        decision = Decision.model_validate(body)
        self.calls += 1
        self.total_cost += decision.usage.cost
        return decision

    def ask(self, state: Any, instructions: str, **criteria: str) -> NoulAnswer:
        """Shorthand for a single yes/no question."""
        answer = self.decide(
            state,
            {"q": NoulQuestion(instructions=instructions, criteria=criteria or None)},
        ).answers["q"]
        assert isinstance(answer, NoulAnswer)
        return answer

    # -- transport ---------------------------------------------------------
    def _send(self, payload: dict[str, Any]) -> tuple[int, Any]:
        if self.cassette is not None and self.cassette.mode == "replay":
            return self.cassette.replay(payload)
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Run in replay mode (the default) "
                "to use the recorded responses in tests/fixtures."
            )
        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        try:
            body = response.json()
        except ValueError:
            body = {"error": {"message": response.text[:500]}}
        if self.cassette is not None and self.cassette.mode == "record":
            self.cassette.record(payload, response.status_code, body)
        return response.status_code, body


def questions_of(kind: str, instructions: str, criteria: Any = None) -> Question:
    """Build a question from plain data (used by the case files)."""
    if kind == "noul":
        return NoulQuestion(instructions=instructions, criteria=criteria)
    if kind == "choice":
        return ChoiceQuestion(instructions=instructions, criteria=criteria)
    if kind == "score":
        return ScoreQuestion(instructions=instructions, criteria=list(criteria or []))
    raise ValueError(f"unknown question type {kind!r}")


def brier_score(pairs: Sequence[tuple[float, bool]]) -> float:
    """Mean squared error of probabilities. 0 = perfect, 0.25 = coin flip."""
    if not pairs:
        return float("nan")
    return sum((p - (1.0 if truth else 0.0)) ** 2 for p, truth in pairs) / len(pairs)
