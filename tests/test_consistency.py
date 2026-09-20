"""Does the same question get the same answer twice?

Anything used for scoring has to be stable, otherwise your metric moves when
your system doesn't. Two things are measured here: repeating an identical
request, and asking the same question with the options reordered or renamed.

The repeat test is marked `live`: replaying it offline would only prove that
the cassette returns what was recorded. Recorded runs are real, though — the
cassette stores the whole sequence, so the numbers below came from five
separate API calls.
"""

from __future__ import annotations

import pytest

from jev_experiments.client import ChoiceQuestion, JevClient, NoulQuestion

REPEATS = 5

AMBIGUOUS = [
    "Oh great, another update that deletes my saved filters. Love that for me.",
    "I'm not angry, just disappointed that the feature I paid for still doesn't work.",
    "Would be nice if the invoices actually matched what I was charged, for once.",
]

TEAMS = {
    "billing": "Payments, charges, invoices, refunds",
    "technical": "Bugs, errors, outages, integrations",
    "account": "Login, password, profile, account settings",
    "sales": "Pricing questions, upgrades, new purchases",
}

TICKETS = [
    ("I was charged twice this month.", "billing"),
    ("The API returns 500 errors on every POST request.", "technical"),
    ("I can't reset my password, the email never arrives.", "account"),
    ("Do you offer discounts for teams of 50+?", "sales"),
    ("Your billing page throws a JavaScript error when I click Save.", "technical"),
    ("I upgraded yesterday but the invoice still shows the old price.", "billing"),
]

COMPLAINT = NoulQuestion(
    instructions="Does this message express a complaint about the product or service?",
    criteria={"true": "A complaint", "false": "Not a complaint"},
)


@pytest.mark.live
def test_repeating_a_question_gives_the_same_verdict(client: JevClient) -> None:
    """Deliberately ambiguous inputs, asked five times each."""
    for text in AMBIGUOUS:
        probabilities = [client.decide(text, {"q": COMPLAINT}).answers["q"].noul for _ in range(REPEATS)]
        spread = max(probabilities) - min(probabilities)
        verdicts = {p > 0.5 for p in probabilities}

        assert len(verdicts) == 1, f"verdict flipped across runs for {text!r}: {probabilities}"
        assert spread < 0.15, f"probability moved by {spread:.2f} across runs: {probabilities}"


def test_answer_does_not_depend_on_option_order(client: JevClient) -> None:
    """Reversing the options must not change which one is picked."""
    reversed_teams = dict(reversed(list(TEAMS.items())))
    for ticket, expected in TICKETS:
        first = client.decide(ticket, {"q": ChoiceQuestion(instructions="Which team should handle this ticket?", criteria=TEAMS)})
        second = client.decide(ticket, {"q": ChoiceQuestion(instructions="Which team should handle this ticket?", criteria=reversed_teams)})
        assert first.answers["q"].choice == second.answers["q"].choice == expected


def test_answer_does_not_depend_on_option_names(client: JevClient) -> None:
    """The meaning lives in the description, not the key.

    Renaming `billing` to `team_a` should change nothing, which is what lets you
    use your own internal codes as option keys.
    """
    renamed = {f"team_{i}": description for i, description in enumerate(TEAMS.values())}
    back = {f"team_{i}": name for i, name in enumerate(TEAMS)}

    for ticket, expected in TICKETS:
        answer = client.decide(
            ticket,
            {"q": ChoiceQuestion(instructions="Which team should handle this ticket?", criteria=renamed)},
        ).answers["q"]
        assert back[answer.choice] == expected
