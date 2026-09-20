"""Scoring an LLM's structured output against a ground-truth record.

This is the experiment that matters if you are evaluating an extraction
pipeline. Exact string matching punishes the model for writing "$1,250.00"
instead of 1250, and cannot pair up list items that came back in a different
order. The evaluator here tries exact matching first and only sends the
genuinely different values to Jev, so most fields cost nothing.

Each case in tests/cases/records.py carries the expected verdict per field
path, so these tests measure the evaluator rather than just exercising it.
"""

from __future__ import annotations

import pytest

from jev_experiments.client import JevClient
from jev_experiments.field_eval import evaluate_dataset, evaluate_record

from tests.cases.enums import ENUM_CASES, ENUM_DEFS
from tests.cases.records import RECORD_PAIRS, RecordCase


def _run(client: JevClient, case: RecordCase):
    # The case files describe an exact-match field in prose; the evaluator wants
    # the rule plus the prose as a hint.
    return evaluate_record(
        case.ground_truth,
        case.prediction,
        client=client,
        rules={path: "exact" for path in case.rules},
        hints={**case.hints, **case.rules},
        match_by=case.match_by,
    )


@pytest.mark.parametrize("case", RECORD_PAIRS, ids=lambda c: c.name)
def test_field_verdicts_match_expectations(client: JevClient, case: RecordCase) -> None:
    """Every leaf path in the case file must get the verdict the case says it should."""
    result = _run(client, case)
    got = {field.path: field.correct for field in result.fields}

    wrong: list[str] = []
    for path, expected in case.expected_correct.items():
        if path not in got:  # a container path such as "items", checked below
            continue
        if got[path] is not expected:
            field = next(f for f in result.fields if f.path == path)
            wrong.append(
                f"{path}: expected {expected}, got {field.correct} "
                f"[{field.method}{'' if field.probability is None else f' p={field.probability:.2f}'}] "
                f"{field.expected!r} vs {field.predicted!r}"
            )

    assert not wrong, f"{case.name}\n  " + "\n  ".join(wrong)


@pytest.mark.parametrize("case", RECORD_PAIRS, ids=lambda c: c.name)
def test_list_level_verdicts(client: JevClient, case: RecordCase) -> None:
    """A container path ("items") means: is the list as a whole right?

    It is wrong if any item is missing, hallucinated, or has a wrong field —
    which is exactly what a purchase order needs.
    """
    containers = {p: v for p, v in case.expected_correct.items() if p.isidentifier() and isinstance(case.ground_truth.get(p), list)}
    if not containers:
        pytest.skip("no container expectations in this case")

    result = _run(client, case)
    for container, expected in containers.items():
        touched = [f for f in result.fields if f.path.startswith(f"{container}[")]
        extras = [e for e in result.extras if e.path.startswith(container)]
        whole_list_ok = all(f.correct for f in touched) and not extras
        assert whole_list_ok is expected, (
            f"{case.name}.{container}: expected {expected}, got {whole_list_ok} "
            f"({sum(not f.correct for f in touched)} bad fields, {len(extras)} extras)"
        )


def test_formatting_differences_are_not_errors(client: JevClient) -> None:
    """The headline: reformatted values pass, changed values fail."""
    case = next(c for c in RECORD_PAIRS if c.name == "invoice_formatting_noise")
    result = _run(client, case)
    by_path = {f.path: f for f in result.fields}

    reformatted = [f for f in result.fields if f.correct and f.method == "jev"]
    assert reformatted, "expected at least one value rescued by semantic matching"

    print(f"\n{case.name}: {result.correct}/{result.total} fields correct")
    for field in result.fields:
        mark = "ok  " if field.correct else "WRONG"
        probability = "" if field.probability is None else f" p={field.probability:.2f}"
        print(f"  {mark} {field.path:<24} [{field.method}{probability}] {field.expected!r} vs {field.predicted!r}")

    # A changed tax rate is a real error, and an ID must match character for character.
    assert by_path["tax_rate"].correct is False
    assert by_path["invoice_id"].correct is False and by_path["invoice_id"].method == "exact"


def test_out_of_vocabulary_enum_values_are_mapped(client: JevClient) -> None:
    """When the model invents a label, map it onto your allowed values.

    Comparing "Computer peripherals" with "Electronics" directly says "different".
    The useful question is which allowed value it belongs to.
    """
    scored = [case for case in ENUM_CASES if case.expected is not None]
    results = []
    for case in scored:
        result = evaluate_record(
            {case.field: case.ground_truth},
            {case.field: case.prediction},
            client=client,
            enums={case.field: ENUM_DEFS[case.field]},
        )
        field = result.fields[0]
        results.append((case, field))

    hits = sum(field.correct is case.expected for case, field in results)
    print(f"\nenum mapping: {hits}/{len(results)} judged as expected")
    for case, field in results:
        if field.correct is not case.expected:
            print(f"  miss: {case.field} {case.ground_truth!r} vs {case.prediction!r} -> mapped to {field.mapped_to!r}")

    assert all(field.out_of_vocabulary for _, field in results), "every prediction here is outside the vocabulary"
    assert hits / len(results) >= 0.85


def test_dataset_report(client: JevClient) -> None:
    """Roll the records up: per-field accuracy, hallucinated extras, review queue."""
    dataset = evaluate_dataset(
        [{"ground_truth": c.ground_truth, "prediction": c.prediction} for c in RECORD_PAIRS],
        client=client,
        rules={path: "exact" for case in RECORD_PAIRS for path in case.rules},
        hints={**{k: v for case in RECORD_PAIRS for k, v in case.hints.items()},
               **{k: v for case in RECORD_PAIRS for k, v in case.rules.items()}},
        match_by={k: v for case in RECORD_PAIRS for k, v in case.match_by.items()},
        concurrency=3,
    )

    print(f"\ndataset: {dataset.overall.correct}/{dataset.overall.total} fields correct "
          f"({dataset.overall.accuracy:.0%}) across {len(dataset.records)} records")
    print(f"  per record: mean {dataset.per_record.mean:.0%}, min {dataset.per_record.min:.0%}, max {dataset.per_record.max:.0%}")
    for stats in dataset.by_field[:5]:
        print(f"  worst field {stats.pattern:<28} {stats.accuracy:.0%} ({stats.correct}/{stats.total})")
    for extra in dataset.extras:
        print(f"  hallucinated {extra.pattern} x{extra.count}")

    assert dataset.overall.total == sum(len(r.fields) for r in dataset.records)
    assert dataset.extras, "the purchase order contains a hallucinated line item"
    # Worst-first ordering is what makes this report useful.
    assert dataset.by_field == sorted(dataset.by_field, key=lambda s: (s.accuracy, -s.total))
