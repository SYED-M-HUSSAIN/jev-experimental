"""Meaning, not wording.

This is the distinction that makes Jev useful for scoring: embedding similarity
would rate "Approved" and "Not approved" as nearly identical, because they share
almost every word. What you need for an evaluation is whether two values say the
same thing, which is a different question.
"""

from __future__ import annotations

import pytest

from jev_experiments.client import JevClient, NoulQuestion

from tests.cases.meaning import DIFFERENT_MEANING, SAME_MEANING

QUESTION = NoulQuestion(
    instructions=(
        "Does `candidate` convey the same meaning as `reference`? Wording, word order "
        "and phrasing may differ; any difference in the actual fact, value, direction "
        "or claim is a mismatch."
    ),
    criteria={"true": "Same meaning", "false": "Different meaning"},
)


def _same(client: JevClient, reference: str, candidate: str) -> float:
    return client.decide({"reference": reference, "candidate": candidate}, {"q": QUESTION}).answers["q"].noul


@pytest.mark.parametrize("reference,candidate", SAME_MEANING, ids=lambda v: v[:28])
def test_different_words_same_meaning(client: JevClient, reference: str, candidate: str) -> None:
    """No shared vocabulary, same fact: "Deceased" / "Passed away"."""
    assert _same(client, reference, candidate) > 0.5


@pytest.mark.parametrize("reference,candidate", DIFFERENT_MEANING, ids=lambda v: v[:28])
def test_similar_words_different_meaning(client: JevClient, reference: str, candidate: str) -> None:
    """Nearly identical text, opposite fact: "Liable" / "Not liable"."""
    assert _same(client, reference, candidate) <= 0.5


def test_overall_agreement_is_high(client: JevClient) -> None:
    """The summary number for the whole set, printed with `pytest -s`."""
    hits = [(_same(client, a, b) > 0.5) for a, b in SAME_MEANING]
    misses = [(_same(client, a, b) <= 0.5) for a, b in DIFFERENT_MEANING]
    correct = sum(hits) + sum(misses)
    total = len(hits) + len(misses)

    print(f"\nmeaning: {correct}/{total} pairs judged correctly "
          f"({sum(hits)}/{len(hits)} paraphrases accepted, {sum(misses)}/{len(misses)} near-misses rejected)")
    assert correct / total >= 0.9
