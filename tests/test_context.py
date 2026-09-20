"""Does the answer follow the context, or just the sentence?

Every case here holds the question fixed and changes only the surrounding
state. If the verdict moves with the context, the model is reading all of it —
which is what lets you hand it a policy, a customer record or a conversation
and get a decision that accounts for them.
"""

from __future__ import annotations

import pytest

from jev_experiments.client import ChoiceQuestion, JevClient, NoulQuestion

from tests.cases.context import CONVERSATION_CONTEXTS, INTENTS, REFUND_CONTEXTS

AUTO_APPROVE = NoulQuestion(
    instructions="Under the stated policy and this customer's history, can this refund be auto-approved?",
    criteria={"true": "Policy allows auto-approval", "false": "Needs human review"},
)
INTENT = ChoiceQuestion(
    instructions="What is the customer agreeing to in their latest message?",
    criteria=INTENTS,
)


@pytest.mark.parametrize("label,state,expected", REFUND_CONTEXTS, ids=lambda v: v if isinstance(v, str) else "")
def test_policy_and_history_change_the_decision(client: JevClient, label: str, state: dict, expected: bool) -> None:
    """Same refund request; different policy limits and customer history."""
    answer = client.decide(state, {"q": AUTO_APPROVE}).answers["q"]
    assert answer.yes is expected, f"{label}: p={answer.noul:.2f}"


@pytest.mark.parametrize("label,state,expected", CONVERSATION_CONTEXTS, ids=lambda v: v if isinstance(v, str) else "")
def test_history_disambiguates_a_bare_reply(client: JevClient, label: str, state: dict, expected: str) -> None:
    """"Yes, go ahead" means nothing on its own; the history decides what it agrees to."""
    answer = client.decide(state, {"q": INTENT}).answers["q"]
    assert answer.choice == expected, f"{label}: got {answer.choice} ({answer.probabilities})"


def test_summary(client: JevClient) -> None:
    """One line per family, printed with `pytest -s`."""
    refunds = [
        (label, client.decide(state, {"q": AUTO_APPROVE}).answers["q"].noul, expected)
        for label, state, expected in REFUND_CONTEXTS
    ]
    print("\nsame refund request, different context:")
    for label, p, expected in refunds:
        print(f"  {label:<34} auto-approve p={p:.2f}  (expected {'yes' if expected else 'no'})")

    print("same reply 'yes, go ahead', different conversation:")
    for label, state, expected in CONVERSATION_CONTEXTS:
        answer = client.decide(state, {"q": INTENT}).answers["q"]
        print(f"  {label:<34} -> {answer.choice:<8} (expected {expected})")
