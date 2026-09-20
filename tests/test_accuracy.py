"""How often is it right, and does 90% actually mean 90%?

Accuracy on its own is not enough for a model you act on automatically. What
matters as much is calibration: if the probabilities are honest, you can
automate the confident cases and route the rest to a person. That threshold is
domain-specific — TypeSafe's own guidance is to validate it on your data — so
treat this file as the shape of the check, not as universal numbers.
"""

from __future__ import annotations

from jev_experiments.client import ChoiceQuestion, JevClient, NoulQuestion, brier_score

from tests.cases.labeled import COMPLAINT_CASES, GROUNDING_CASES, ROUTING_CASES, TEAMS, NoulCase

ALL_NOUL: list[NoulCase] = [*COMPLAINT_CASES, *GROUNDING_CASES]


# The criteria carry the meaning, not the question key. Left vaguer than this
# ("true": "Yes" / "false": "No"), the complaint cases lose ~2 of the hard ones:
# a customer griping about their OWN VPN, and a feature request phrased as a
# mild gripe, both get read as complaints. Defining the boundary fixes them.
COMPLAINT_CRITERIA = {
    "true": "Dissatisfaction with our product, service, billing or support",
    "false": (
        "Anything else: praise, a neutral question, a feature request or wish, "
        "or dissatisfaction aimed at something other than us (their own systems, "
        "another vendor, a past provider)"
    ),
}
GROUNDING_CRITERIA = {
    "true": "Every claim follows from the documents, including any arithmetic",
    "false": "A claim contradicts the documents, is not stated in them, or the arithmetic is wrong",
}


def _ask(client: JevClient, case: NoulCase) -> float:
    criteria = COMPLAINT_CRITERIA if case in COMPLAINT_CASES else GROUNDING_CRITERIA
    question = NoulQuestion(instructions=case.question, criteria=criteria)
    return client.decide(case.state, {"q": question}).answers["q"].noul


def _accuracy(results: list[tuple[NoulCase, float]]) -> float:
    return sum((p > 0.5) == c.expected for c, p in results) / len(results)


def test_yes_no_accuracy(client: JevClient) -> None:
    """Easy cases should be near-perfect; hard ones are the interesting number."""
    results = [(case, _ask(client, case)) for case in ALL_NOUL]
    easy = [r for r in results if not r[0].hard]
    hard = [r for r in results if r[0].hard]

    print(
        f"\nyes/no: overall {_accuracy(results):.0%} (n={len(results)}) · "
        f"easy {_accuracy(easy):.0%} (n={len(easy)}) · hard {_accuracy(hard):.0%} (n={len(hard)})"
    )
    for case, p in results:
        if (p > 0.5) != case.expected:
            print(f"  miss: expected {case.expected} got p={p:.2f} · {str(case.state)[:70]}")

    assert _accuracy(results) >= 0.85
    assert _accuracy(easy) >= 0.95


def test_choice_accuracy(client: JevClient) -> None:
    """Routing: the surface topic is sometimes a trap (a JS error on the billing page)."""
    question = ChoiceQuestion(instructions="Which team should handle this ticket?", criteria=TEAMS)
    answers = [client.decide(case.state, {"q": question}).answers["q"] for case in ROUTING_CASES]
    hits = sum(a.choice == c.expected for a, c in zip(answers, ROUTING_CASES))

    print(f"routing: {hits}/{len(ROUTING_CASES)} correct")
    for answer, case in zip(answers, ROUTING_CASES):
        if answer.choice != case.expected:
            print(f"  miss: expected {case.expected} got {answer.choice} (conf {answer.confidence:.2f}) · {str(case.state)[:60]}")

    assert hits / len(ROUTING_CASES) >= 0.85


def test_probabilities_are_calibrated(client: JevClient) -> None:
    """Brier score plus confidence buckets: does claimed confidence match reality?"""
    results = [(case, _ask(client, case)) for case in ALL_NOUL]
    score = brier_score([(p, c.expected) for c, p in results])

    buckets = {
        "very confident (>=90%)": lambda p: p >= 0.9 or p <= 0.1,
        "confident (70-90%)": lambda p: 0.7 <= p < 0.9 or 0.1 < p <= 0.3,
        "unsure (30-70%)": lambda p: 0.3 < p < 0.7,
    }
    print(f"\ncalibration: Brier {score:.3f} (0 = perfect, 0.25 = coin flip)")
    for label, in_bucket in buckets.items():
        chosen = [(c, p) for c, p in results if in_bucket(p)]
        if not chosen:
            print(f"  {label:<24} no cases")
            continue
        claimed = sum(max(p, 1 - p) for _, p in chosen) / len(chosen)
        actual = _accuracy(chosen)
        print(f"  {label:<24} n={len(chosen):<3} claims {claimed:.0%}, right {actual:.0%}")

    assert score < 0.10, "probabilities are not usefully calibrated"

    # The practical payoff: mistakes should cluster in the unsure band, not be
    # asserted confidently. That is what makes a review threshold work.
    confident_misses = [(c, p) for c, p in results if (p >= 0.9 or p <= 0.1) and (p > 0.5) != c.expected]
    assert len(confident_misses) <= 1, f"confidently wrong on {len(confident_misses)} cases"
