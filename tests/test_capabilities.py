"""What the model can and cannot be asked.

This is the experiment behind the claim people most often get wrong: Jev is not
a small LLM. It answers three kinds of typed question and refuses everything
else, including the one thing most people assume it does — extraction.
"""

from __future__ import annotations

import pytest

from jev_experiments.client import (
    ChoiceAnswer,
    ChoiceQuestion,
    JevAPIError,
    JevClient,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

TICKET = (
    "Hi, I've been trying to connect my payment account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)


def test_the_three_question_types_answer_one_state(client: JevClient) -> None:
    """All three types can be asked about the same state in a single request."""
    result = client.decide(
        TICKET,
        {
            "is_urgent": NoulQuestion(
                instructions="Does this message convey urgency or time-sensitivity?",
                criteria={"true": "Time-sensitive", "false": "Can wait"},
            ),
            "team": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={
                    "billing": "Payments, invoices, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            ),
            "frustration": ScoreQuestion(
                instructions="How frustrated is the customer?",
                criteria=["Calm", "Frustrated", "Very angry"],
            ),
        },
    )

    urgent, team, frustration = (result.answers[k] for k in ("is_urgent", "team", "frustration"))
    assert isinstance(urgent, NoulAnswer) and urgent.yes
    assert isinstance(team, ChoiceAnswer) and team.choice in {"billing", "technical", "sales"}
    assert isinstance(frustration, ScoreAnswer) and 0.0 <= frustration.score <= 2.0

    # A noul is a probability, a choice is a distribution over the options you defined.
    assert 0.0 <= urgent.noul <= 1.0
    assert abs(sum(team.probabilities.values()) - 1.0) < 0.02
    assert set(team.probabilities) == {"billing", "technical", "sales"}


def test_it_cannot_extract_a_value(client: JevClient) -> None:
    """The headline limitation: there is no question type that returns a value.

    Ask for a string and the API rejects the request, naming the only three
    types it accepts. If you need the invoice total, you still need an LLM.
    """
    with pytest.raises(JevAPIError) as caught:
        client.decide(
            "Invoice #4521 from a supplier, total $1,250.00, due Oct 3.",
            {"amount": {"type": "string", "instructions": "Extract the invoice total"}},
        )

    assert caught.value.status == 400
    message = caught.value.message
    assert "noul" in message and "choice" in message and "score" in message


def test_state_can_be_nested_json(client: JevClient) -> None:
    """State is not limited to a string: pass the structure you already have."""
    answer = client.ask(
        {
            "order": {"id": "A-1187", "total": 420.00, "status": "shipped"},
            "customer": {"plan": "enterprise", "open_tickets": 3},
            "message": "Where is my order? It was supposed to arrive on Tuesday.",
        },
        "Is the customer asking about the delivery status of an existing order?",
        **{"true": "Asking about delivery of an existing order", "false": "Something else"},
    )
    assert answer.yes


def test_many_questions_cost_little_more_than_one(client: JevClient) -> None:
    """Questions are evaluated in parallel; extra ones only add their own tokens.

    This is why batching field comparisons into one request is the right shape:
    the state is sent once and every question sees it.
    """
    state = {
        "email": "Please cancel my subscription and confirm in writing. I've already been charged twice.",
    }
    one = client.decide(
        state,
        {"a": NoulQuestion(instructions="Does the sender want to cancel their subscription?")},
    )
    many = client.decide(
        state,
        {
            "a": NoulQuestion(instructions="Does the sender want to cancel their subscription?"),
            "b": NoulQuestion(instructions="Does the sender report a billing problem?"),
            "c": NoulQuestion(instructions="Does the sender ask for written confirmation?"),
            "d": NoulQuestion(instructions="Is the sender angry?"),
            "e": NoulQuestion(instructions="Does the message mention a refund amount?"),
        },
    )

    assert len(many.answers) == 5
    # Same state, same verdict, whether asked alone or alongside four others.
    assert (one.answers["a"].noul > 0.5) == (many.answers["a"].noul > 0.5)
    # Five questions cost far less than five separate requests would.
    assert many.usage.cost < one.usage.cost * 2.5
