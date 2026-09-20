"""Is this claim actually in the document?

The most valuable use found in these experiments. Comparing a model's output
with a reference answer only tells you the two disagree; checking both against
the source tells you which one is right — and catches values the model invented
that no reference comparison can see.

It also separates "contradicted" from "not mentioned", which matters: one is a
wrong answer, the other is a fabrication.
"""

from __future__ import annotations

import pytest

from jev_experiments.client import ChoiceQuestion, JevClient, NoulQuestion

from tests.cases.grounding import CLAIM_CASES, DOCS, ClaimCase

VERDICTS = {
    "supported": "The document states this, or it follows directly from what the document states",
    "contradicted": "The document states something different",
    "not_mentioned": "The document does not address this at all",
}


def _classify(client: JevClient, case: ClaimCase):
    question = ChoiceQuestion(
        instructions=(
            "Judge the claim against the document only. Wording may differ and simple "
            "unit conversions are fine; a different number, date or fact is a contradiction."
        ),
        criteria=VERDICTS,
    )
    return client.decide({"document": DOCS[case.doc], "claim": case.claim}, {"q": question}).answers["q"]


@pytest.mark.parametrize("case", CLAIM_CASES, ids=lambda c: c.claim[:34])
def test_claim_verdicts(client: JevClient, case: ClaimCase) -> None:
    answer = _classify(client, case)
    assert answer.choice == case.verdict, f"{case.claim!r} -> {answer.choice} ({answer.probabilities})"


def test_supported_vs_invented(client: JevClient) -> None:
    """Summary table, and the number that matters: nothing invented slips through."""
    results = [(case, _classify(client, case)) for case in CLAIM_CASES]
    hits = sum(answer.choice == case.verdict for case, answer in results)

    print(f"\ngrounding: {hits}/{len(results)} claims classified correctly")
    for case, answer in results:
        mark = "ok  " if answer.choice == case.verdict else "MISS"
        print(f"  {mark} [{case.doc}] {case.claim[:58]:<58} {answer.choice} (expected {case.verdict})")

    assert hits / len(results) >= 0.85

    # A fabricated claim must never be reported as supported: that is the failure
    # mode this check exists to catch.
    invented = [(c, a) for c, a in results if c.verdict == "not_mentioned"]
    assert all(a.choice != "supported" for _, a in invented)


def test_yes_no_form_for_pipelines(client: JevClient) -> None:
    """The same check as a single yes/no, which is what you batch at scale."""
    question = NoulQuestion(
        instructions="Is this claim supported by the document?",
        criteria={
            "true": "The document supports it, allowing for rewording and unit conversion",
            "false": "The document contradicts it or does not mention it",
        },
    )
    for case in CLAIM_CASES:
        answer = client.decide({"document": DOCS[case.doc], "claim": case.claim}, {"q": question}).answers["q"]
        assert answer.yes is (case.verdict == "supported"), f"{case.claim!r} p={answer.noul:.2f}"
