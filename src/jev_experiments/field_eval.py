"""Field-level evaluation: compare an LLM's output against ground truth, key by key.

The point of this module is to spend as few Jev calls as possible while still
being fair to the model. Most fields in a structured extraction are either
missing or byte-identical, and neither case needs a judge. Only the genuinely
ambiguous leaves ("St." vs "Street", "2024-01-05" vs "Jan 5 2024") are sent to
Jev, batched into as few requests as possible.

    Objects           -> recurse into every key; paths like "customer.address.city"
    Arrays of objects -> line up ground-truth items with predicted items (by a key
                         field, or by asking Jev which item corresponds), then
                         recurse: "items[0].qty". Unmatched ground-truth items count
                         as missing; unmatched predicted items are reported as extras.
    Leaves            -> strings, numbers, booleans, None, arrays of primitives:
        1. Missing                   -> incorrect (no API call)
        2. Equal after normalisation -> correct   (no API call)
        3. Otherwise                 -> Jev judges semantic equivalence (batched)

Enum fields (`enums`): if the predicted value isn't one of the allowed values,
Jev maps it onto the closest allowed value (or "none"), and the *mapped* value is
compared with the ground truth. The field is flagged `out_of_vocabulary` either
way, because the model broke the schema. With `enum_policy="strict"` an
out-of-vocabulary value simply counts as wrong, with no Jev call.

`rules`, `hints`, `enums` and `match_by` are keyed by *path pattern*, which is a
path with array indexes written as "[]": e.g. "items[].sku",
"customer.address.zip". Actual paths keep their indexes: "items[0].sku".
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel, Field

from .client import ChoiceAnswer, ChoiceQuestion, JevClient, NoulAnswer, NoulQuestion

__all__ = [
    "DatasetResult",
    "EvalOptions",
    "FieldResult",
    "FieldRule",
    "FieldStats",
    "RecordPair",
    "RecordResult",
    "SectionStats",
    "evaluate_dataset",
    "evaluate_record",
]

FieldRule = Literal["semantic", "exact"]
EnumPolicy = Literal["map", "strict"]
Method = Literal["missing", "exact", "jev", "enum-map", "pending"]

#: Allowed values for an enum pattern: a plain list, or {value: description}.
EnumSpec = Sequence[str] | Mapping[str, str]


class _Missing:
    """Absence of a key, kept distinct from an explicit ``None`` in the data."""

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<missing>"


MISSING = _Missing()

_SECTION_SPLIT = re.compile(r"[.\[]")
_INDEX = re.compile(r"\[\+?\d+\]")


# --------------------------------------------------------------------------- results
class FieldResult(BaseModel):
    """The verdict on a single leaf value, plus how that verdict was reached."""

    path: str  # e.g. "items[1].qty"
    pattern: str  # e.g. "items[].qty", used for aggregation
    expected: Any = None
    predicted: Any = None
    correct: bool
    method: Method  # "pending" = dry run; this one would have gone to Jev
    probability: float | None = None  # Jev's p(match), or p(ground truth) when mapping enums
    uncertain: bool | None = None  # Jev was not confident either way
    out_of_vocabulary: bool | None = None  # enum field whose value is not an allowed value
    mapped_to: str | None = None  # allowed value Jev mapped it onto (None = none fit)


class Extra(BaseModel):
    """A key or list item present in the prediction but absent from the ground truth."""

    path: str
    value: Any = None


class SectionStats(BaseModel):
    correct: int = 0
    total: int = 0
    accuracy: float = 0.0


class RecordResult(BaseModel):
    """Everything learned about one (ground truth, prediction) pair."""

    fields: list[FieldResult] = Field(default_factory=list)
    extras: list[Extra] = Field(default_factory=list)
    by_section: dict[str, SectionStats] = Field(default_factory=dict)
    correct: int = 0
    total: int = 0
    accuracy: float = 0.0
    cost: float = 0.0


class FieldStats(BaseModel):
    """Dataset-wide counts for one path pattern (or one top-level section)."""

    pattern: str  # e.g. "items[].qty"
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    missing: int = 0  # absent from the prediction
    jev_judged: int = 0  # needed a semantic judgement
    uncertain: int = 0  # Jev's answer landed inside the review band
    out_of_vocabulary: int = 0  # enum values outside the allowed list


class Overall(BaseModel):
    correct: int = 0
    total: int = 0
    accuracy: float = 0.0


class PerRecord(BaseModel):
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0


class ExtraCount(BaseModel):
    pattern: str
    count: int


class DatasetResult(BaseModel):
    """Aggregate view over many records: what the model gets wrong, and where."""

    records: list[RecordResult] = Field(default_factory=list)
    overall: Overall = Field(default_factory=Overall)
    per_record: PerRecord = Field(default_factory=PerRecord)
    by_field: list[FieldStats] = Field(default_factory=list)  # worst accuracy first
    by_section: list[FieldStats] = Field(default_factory=list)  # top-level keys, worst first
    extras: list[ExtraCount] = Field(default_factory=list)  # most frequent first
    uncertain: list[FieldResult] = Field(default_factory=list)  # fields worth a human look
    cost: float = 0.0


# --------------------------------------------------------------------------- options
class EvalOptions(BaseModel):
    """Every knob of the comparison, bundled so a config can be reused verbatim.

    Each field is also accepted as a keyword argument on `evaluate_record` and
    `evaluate_dataset`; an explicit keyword wins over the value in the bundle.
    """

    model_config = {"extra": "forbid"}

    rules: dict[str, FieldRule] = Field(default_factory=dict)
    """Per path pattern; default "semantic". "exact" never calls Jev."""

    hints: dict[str, str] = Field(default_factory=dict)
    """Per path pattern: extra matching guidance appended to the question."""

    match_by: dict[str, str] = Field(default_factory=dict)
    """Array pattern -> item key used to line items up, e.g. {"items": "sku"}."""

    threshold: float = 0.5
    """p(match) strictly above this counts as correct."""

    review: tuple[float, float] = (0.3, 0.7)
    """Jev answers inside this band are flagged uncertain."""

    batch_size: int = 20
    """Leaf questions per Jev call."""

    enums: dict[str, EnumSpec] = Field(default_factory=dict)
    """Pattern -> allowed values, as a list or {value: description}."""

    enum_policy: EnumPolicy = "map"
    """Out-of-vocabulary values: map onto the closest allowed value, or count as wrong."""

    dry_run: bool = False
    """No API calls: lists pair by index, fields needing Jev are marked "pending"."""


class RecordPair(BaseModel):
    """One evaluation unit. Also accepted as a 2-tuple or a plain dict."""

    ground_truth: dict[str, Any]
    prediction: dict[str, Any]


RecordLike = RecordPair | Mapping[str, Any] | tuple[Mapping[str, Any], Mapping[str, Any]]


def _resolve_options(options: EvalOptions | None, overrides: Mapping[str, Any]) -> EvalOptions:
    """Merge explicit keyword arguments over an optional `EvalOptions` bundle.

    Keeping both forms lets callers write a throwaway `dry_run=True` check as
    well as ship a reusable, serialisable config, without two code paths.
    """
    base = options.model_copy(deep=True) if options is not None else EvalOptions()
    given = {key: value for key, value in overrides.items() if value is not None}
    return base.model_copy(update=given) if given else base


def _as_pair(record: RecordLike) -> RecordPair:
    """Accept a `RecordPair`, a (ground_truth, prediction) tuple, or a dict of either naming."""
    if isinstance(record, RecordPair):
        return record
    if isinstance(record, tuple):
        ground_truth, prediction = record
        return RecordPair(ground_truth=dict(ground_truth), prediction=dict(prediction))
    if "ground_truth" in record or "prediction" in record:
        return RecordPair(
            ground_truth=dict(record.get("ground_truth") or {}),
            prediction=dict(record.get("prediction") or {}),
        )
    if "groundTruth" in record:  # tolerate the TypeScript spelling
        return RecordPair(
            ground_truth=dict(record.get("groundTruth") or {}),
            prediction=dict(record.get("prediction") or {}),
        )
    raise TypeError(f"cannot read a (ground_truth, prediction) pair from {record!r}")


# --------------------------------------------------------------------------- helpers
def _is_object(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_empty(value: Any) -> bool:
    """True for the several ways a record says "no value here".

    Ground truth often stores null where a model returns an empty string; both
    mean the field was absent, so counting that as a mismatch is noise.
    """
    return value is None or isinstance(value, _Missing) or (isinstance(value, str) and not value.strip())


def _normalize(value: Any) -> str:
    """Render a value as the string used for "is this literally the same?" checks.

    Strings lose case and surrounding/repeated whitespace, because no model
    should be marked wrong for capitalising a city. Everything else is dumped as
    canonical JSON so that dicts and lists compare structurally.
    """
    if isinstance(value, _Missing):
        return "\x00missing"
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip().lower())
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, float) and value.is_integer():
        return json.dumps(int(value))  # 3.0 and 3 are the same number
    return json.dumps(value, sort_keys=True, default=str)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _section_of(path: str) -> str:
    return _SECTION_SPLIT.split(path, maxsplit=1)[0]


class _Leaf(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    path: str
    pattern: str
    expected: Any = None
    predicted: Any = None
    present: bool


# --------------------------------------------------------------------------- record
def evaluate_record(
    ground_truth: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    client: JevClient | None = None,
    options: EvalOptions | None = None,
    rules: Mapping[str, FieldRule] | None = None,
    hints: Mapping[str, str] | None = None,
    match_by: Mapping[str, str] | None = None,
    threshold: float | None = None,
    review: tuple[float, float] | None = None,
    batch_size: int | None = None,
    enums: Mapping[str, EnumSpec] | None = None,
    enum_policy: EnumPolicy | None = None,
    dry_run: bool | None = None,
) -> RecordResult:
    """Score one prediction against one ground-truth record, field by field.

    A single accuracy number hides which part of the schema the model is bad at,
    so this returns a verdict per leaf path along with the method used to reach
    it. That makes a regression traceable to "items[].qty went to Jev and lost"
    rather than "accuracy dropped two points".
    """
    return _Evaluator(
        _resolve_options(
            options,
            {
                "rules": rules,
                "hints": hints,
                "match_by": match_by,
                "threshold": threshold,
                "review": review,
                "batch_size": batch_size,
                "enums": enums,
                "enum_policy": enum_policy,
                "dry_run": dry_run,
            },
        ),
        client,
    ).run(ground_truth, prediction)


class _Evaluator:
    """One record's walk-then-judge pass. Holds the leaves, extras and cost."""

    def __init__(self, options: EvalOptions, client: JevClient | None) -> None:
        self.opt = options
        self._client = client
        self.leaves: list[_Leaf] = []
        self.extras: list[Extra] = []
        self.cost = 0.0

    @property
    def client(self) -> JevClient:
        """A client is only needed once something actually has to be judged."""
        if self._client is None:
            self._client = JevClient()
        return self._client

    def run(self, ground_truth: Mapping[str, Any], prediction: Mapping[str, Any]) -> RecordResult:
        self._walk(ground_truth, prediction, "", "", present=True)
        fields = self._score_leaves()
        return self._summarize(fields)

    # -- 1. walk ground truth, pairing each node with its prediction ---------
    def _walk(self, gt: Any, pred: Any, path: str, pattern: str, *, present: bool) -> None:
        if _is_object(gt) and len(gt) > 0:
            p: Mapping[str, Any] = pred if _is_object(pred) else {}
            for key in gt:
                self._walk(
                    gt[key],
                    p.get(key, MISSING),
                    _join(path, key),
                    _join(pattern, key),
                    present=present and key in p,
                )
            for key in p:
                if key not in gt:
                    self.extras.append(Extra(path=_join(path, key), value=p[key]))
            return

        if isinstance(gt, list) and gt and all(_is_object(item) for item in gt):
            pred_items: list[Any] = pred if isinstance(pred, list) else []
            pairs = self._align_arrays(gt, pred_items, path, self.opt.match_by.get(pattern))
            used = {j for j in pairs if j is not None}
            for i, item in enumerate(gt):
                j = pairs[i]
                self._walk(
                    item,
                    MISSING if j is None else pred_items[j],
                    f"{path}[{i}]",
                    f"{pattern}[]",
                    present=present and j is not None,
                )
            for j, item in enumerate(pred_items):
                if j not in used:
                    self.extras.append(Extra(path=f"{path}[+{j}]", value=item))
            return

        self.leaves.append(
            _Leaf(
                path=path,
                pattern=pattern,
                expected=gt,
                predicted=pred,
                present=present and not isinstance(pred, _Missing),
            )
        )

    # -- 2. line up array items: by key field, else ask Jev ------------------
    def _align_arrays(
        self,
        gt: Sequence[Mapping[str, Any]],
        pred: Sequence[Any],
        path: str,
        key: str | None,
    ) -> list[int | None]:
        """Decide which predicted item corresponds to each ground-truth item.

        Position is not identity: a model may list the same three line items in
        another order. So we pair on a stable key when the schema has one, and
        otherwise ask Jev about identity only, before comparing any details.
        """
        pairs: list[int | None] = [None] * len(gt)
        if not pred:
            return pairs

        if key:
            used: set[int] = set()
            for i, item in enumerate(gt):
                want = _normalize(item.get(key, MISSING))
                for j, candidate in enumerate(pred):
                    if j in used or not _is_object(candidate):
                        continue
                    if _normalize(candidate.get(key, MISSING)) == want:
                        pairs[i] = j
                        used.add(j)
                        break
            return pairs

        if len(gt) == 1 and len(pred) == 1:
            return [0]
        if self.opt.dry_run:
            return [i if i < len(pred) else None for i in range(len(gt))]

        criteria = {"none": "No predicted item corresponds to this one"}
        criteria.update({f"p{j}": f"predicted[{j}]" for j in range(len(pred))})
        questions = {
            f"g{i}": ChoiceQuestion(
                instructions=(
                    f'Which item in "predicted" refers to the same real-world entity as '
                    f"expected[{i}]? Judge by identity (what the item is), not by whether "
                    f"its details are correct."
                ),
                criteria=criteria,
            )
            for i in range(len(gt))
        }
        decision = self.client.decide(
            {"list": path, "expected": list(gt), "predicted": list(pred)}, questions
        )
        self.cost += decision.usage.cost

        # Greedy one-to-one assignment, most confident first.
        candidates: list[tuple[float, int, int, float]] = []
        for i in range(len(gt)):
            answer = decision.answers[f"g{i}"]
            assert isinstance(answer, ChoiceAnswer)
            none_p = answer.probabilities.get("none", 0.0)
            for option, p in answer.probabilities.items():
                if option != "none":
                    candidates.append((p, i, int(option[1:]), none_p))
        candidates.sort(key=lambda c: c[0], reverse=True)

        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for p, i, j, none_p in candidates:
            if i in used_gt or j in used_pred or p <= none_p or p < 0.3:
                continue
            pairs[i] = j
            used_gt.add(i)
            used_pred.add(j)
        return pairs

    # -- 3. score leaves: missing / exact / enum / Jev -----------------------
    def _allowed_values(self, pattern: str) -> list[str] | None:
        spec = self.opt.enums.get(pattern)
        if spec is None:
            return None
        return list(spec.keys()) if isinstance(spec, Mapping) else list(spec)

    def _score_leaves(self) -> list[FieldResult]:
        fields: list[FieldResult] = []
        to_judge: list[_Leaf] = []
        to_map: list[_Leaf] = []

        for leaf in self.leaves:
            allowed = self._allowed_values(leaf.pattern)
            out_of_vocab = bool(
                allowed is not None
                and leaf.present
                and leaf.predicted is not None
                and not any(_normalize(v) == _normalize(leaf.predicted) for v in allowed)
            )
            if not leaf.present:
                fields.append(self._verdict(leaf, correct=_is_empty(leaf.expected), method="missing"))
            elif _is_empty(leaf.expected) and _is_empty(leaf.predicted):
                # null, "" and "   " all mean "the field was not filled in".
                fields.append(self._verdict(leaf, correct=True, method="exact"))
            elif out_of_vocab and self.opt.enum_policy == "strict":
                fields.append(
                    self._verdict(leaf, correct=False, method="exact", out_of_vocabulary=True)
                )
            elif out_of_vocab:
                to_map.append(leaf)
            elif _normalize(leaf.expected) == _normalize(leaf.predicted):
                fields.append(self._verdict(leaf, correct=True, method="exact"))
            elif allowed is not None:
                # Both values are legal enum members but differ: no judgement needed.
                fields.append(self._verdict(leaf, correct=False, method="exact"))
            elif (
                self.opt.rules.get(leaf.pattern) == "exact"
                or leaf.expected is None
                or leaf.predicted is None
            ):
                fields.append(self._verdict(leaf, correct=False, method="exact"))
            else:
                to_judge.append(leaf)

        if self.opt.dry_run:
            fields += [
                self._verdict(leaf, correct=False, method="pending", out_of_vocabulary=True)
                for leaf in to_map
            ]
            fields += [self._verdict(leaf, correct=False, method="pending") for leaf in to_judge]
            return fields

        fields += self._map_enum_values(to_map)
        fields += self._judge_semantically(to_judge)
        return fields

    def _verdict(
        self,
        leaf: _Leaf,
        *,
        correct: bool,
        method: Method,
        out_of_vocabulary: bool | None = None,
        **extra: Any,
    ) -> FieldResult:
        return FieldResult(
            path=leaf.path,
            pattern=leaf.pattern,
            expected=leaf.expected,
            predicted=None if isinstance(leaf.predicted, _Missing) else leaf.predicted,
            correct=correct,
            method=method,
            out_of_vocabulary=out_of_vocabulary,
            **extra,
        )

    def _batches(self, leaves: Sequence[_Leaf]) -> list[Sequence[_Leaf]]:
        size = max(1, self.opt.batch_size)
        return [leaves[start : start + size] for start in range(0, len(leaves), size)]

    def _map_enum_values(self, leaves: Sequence[_Leaf]) -> list[FieldResult]:
        """Ask Jev which allowed value an off-schema answer actually meant.

        A model that answers "USD $" for a currency enum is wrong about the
        schema but right about the world; mapping first separates a formatting
        failure from a factual one, and both stay visible via
        `out_of_vocabulary`.
        """
        results: list[FieldResult] = []
        for batch in self._batches(leaves):
            state: dict[str, Any] = {}
            questions: dict[str, ChoiceQuestion] = {}
            for i, leaf in enumerate(batch):
                allowed = self._allowed_values(leaf.pattern) or []
                spec = self.opt.enums.get(leaf.pattern)
                criteria = {
                    f"v{j}": (f"{v}: {spec[v]}" if isinstance(spec, Mapping) else v)
                    for j, v in enumerate(allowed)
                }
                criteria["none"] = "None of the allowed values means the same thing"
                state[leaf.path] = leaf.predicted
                hint = self.opt.hints.get(leaf.pattern)
                questions[f"m{i}"] = ChoiceQuestion(
                    instructions=(
                        f'The value at "{leaf.path}" should have been one of the allowed '
                        f"options but isn't. Which allowed option has the same meaning as "
                        f'this value? Pick "none" if it is unrelated, ambiguous between '
                        f"several options, or broader/narrower than any single option."
                        + (f" {hint}" if hint else "")
                    ),
                    criteria=criteria,
                )

            decision = self.client.decide(state, questions)
            self.cost += decision.usage.cost

            for i, leaf in enumerate(batch):
                allowed = self._allowed_values(leaf.pattern) or []
                answer = decision.answers[f"m{i}"]
                assert isinstance(answer, ChoiceAnswer)
                mapped_to = None if answer.choice == "none" else allowed[int(answer.choice[1:])]
                expected_key = next(
                    (j for j, v in enumerate(allowed) if _normalize(v) == _normalize(leaf.expected)),
                    None,
                )
                results.append(
                    self._verdict(
                        leaf,
                        correct=mapped_to is not None
                        and _normalize(mapped_to) == _normalize(leaf.expected),
                        method="enum-map",
                        out_of_vocabulary=True,
                        probability=(
                            answer.probabilities.get(f"v{expected_key}", 0.0)
                            if expected_key is not None
                            else 0.0
                        ),
                        uncertain=answer.confidence < self.opt.review[1],
                        mapped_to=mapped_to,
                    )
                )
        return results

    def _judge_semantically(self, leaves: Sequence[_Leaf]) -> list[FieldResult]:
        """Ask Jev whether two differing values carry the same information.

        Batched, because the Decisions API answers questions in one request in
        parallel: twenty leaves cost barely more wall-clock time than one.
        """
        results: list[FieldResult] = []
        for batch in self._batches(leaves):
            state: dict[str, Any] = {}
            questions: dict[str, NoulQuestion] = {}
            for i, leaf in enumerate(batch):
                state[leaf.path] = {"ground_truth": leaf.expected, "prediction": leaf.predicted}
                hint = self.opt.hints.get(leaf.pattern)
                questions[f"q{i}"] = NoulQuestion(
                    instructions=(
                        f'For field "{leaf.path}", does the prediction convey the same '
                        f"information as the ground_truth? Ignore formatting, casing, "
                        f"abbreviations, list order and equivalent units or date formats; "
                        f"any difference in the actual value, name, quantity or meaning is "
                        f"a mismatch." + (f" {hint}" if hint else "")
                    ),
                    criteria={"true": "Same information", "false": "Different or incorrect information"},
                )

            decision = self.client.decide(state, questions)
            self.cost += decision.usage.cost

            low, high = self.opt.review
            for i, leaf in enumerate(batch):
                answer = decision.answers[f"q{i}"]
                assert isinstance(answer, NoulAnswer)
                p = answer.noul
                results.append(
                    self._verdict(
                        leaf,
                        correct=p > self.opt.threshold,
                        method="jev",
                        probability=p,
                        uncertain=low < p < high,
                    )
                )
        return results

    # -- 4. summarize -------------------------------------------------------
    def _summarize(self, fields: list[FieldResult]) -> RecordResult:
        order = {leaf.path: i for i, leaf in enumerate(self.leaves)}
        fields.sort(key=lambda f: order.get(f.path, len(order)))

        by_section: dict[str, SectionStats] = {}
        for field in fields:
            stats = by_section.setdefault(_section_of(field.path), SectionStats())
            stats.total += 1
            stats.correct += field.correct
            stats.accuracy = stats.correct / stats.total

        correct = sum(f.correct for f in fields)
        return RecordResult(
            fields=fields,
            extras=self.extras,
            by_section=by_section,
            correct=correct,
            total=len(fields),
            accuracy=correct / len(fields) if fields else 0.0,
            cost=self.cost,
        )


# --------------------------------------------------------------------------- dataset
def evaluate_dataset(
    records: Sequence[RecordLike],
    *,
    client: JevClient | None = None,
    options: EvalOptions | None = None,
    concurrency: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
    rules: Mapping[str, FieldRule] | None = None,
    hints: Mapping[str, str] | None = None,
    match_by: Mapping[str, str] | None = None,
    threshold: float | None = None,
    review: tuple[float, float] | None = None,
    batch_size: int | None = None,
    enums: Mapping[str, EnumSpec] | None = None,
    enum_policy: EnumPolicy | None = None,
    dry_run: bool | None = None,
) -> DatasetResult:
    """Evaluate many records and roll the results up into one diagnosis.

    The useful output is not the headline accuracy but the ranking underneath
    it: which path patterns fail most often, which sections drag the average
    down, which keys the model invents, and which verdicts Jev itself was
    unsure about and a human should re-check.

    Records run on a thread pool because the Jev client is synchronous
    `requests`; the work is all network wait, so threads are enough.
    """
    resolved = _resolve_options(
        options,
        {
            "rules": rules,
            "hints": hints,
            "match_by": match_by,
            "threshold": threshold,
            "review": review,
            "batch_size": batch_size,
            "enums": enums,
            "enum_policy": enum_policy,
            "dry_run": dry_run,
        },
    )
    pairs = [_as_pair(record) for record in records]
    if not pairs:
        return DatasetResult()

    shared_client = client if client is not None or resolved.dry_run else JevClient()
    done = 0
    lock = threading.Lock()

    def evaluate(pair: RecordPair) -> RecordResult:
        nonlocal done
        result = evaluate_record(
            pair.ground_truth, pair.prediction, client=shared_client, options=resolved
        )
        if on_progress is not None:
            with lock:
                done += 1
                on_progress(done, len(pairs))
        return result

    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(pairs)))) as pool:
        results = list(pool.map(evaluate, pairs))

    fields = [f for result in results for f in result.fields]

    def group(key: Callable[[FieldResult], str]) -> list[FieldStats]:
        grouped: dict[str, FieldStats] = {}
        for field in fields:
            k = key(field)
            stats = grouped.setdefault(k, FieldStats(pattern=k))
            stats.total += 1
            stats.correct += field.correct
            stats.missing += field.method == "missing"
            stats.jev_judged += field.method in ("jev", "enum-map")
            stats.uncertain += bool(field.uncertain)
            stats.out_of_vocabulary += bool(field.out_of_vocabulary)
            stats.accuracy = stats.correct / stats.total
        return sorted(grouped.values(), key=lambda s: (s.accuracy, -s.total))

    extra_counts: dict[str, int] = {}
    for result in results:
        for extra in result.extras:
            pattern = _INDEX.sub("[]", extra.path)
            extra_counts[pattern] = extra_counts.get(pattern, 0) + 1

    accuracies = [r.accuracy for r in results]
    correct = sum(f.correct for f in fields)
    return DatasetResult(
        records=results,
        overall=Overall(
            correct=correct,
            total=len(fields),
            accuracy=correct / len(fields) if fields else 0.0,
        ),
        per_record=PerRecord(
            mean=sum(accuracies) / len(accuracies),
            min=min(accuracies),
            max=max(accuracies),
        ),
        by_field=group(lambda f: f.pattern),
        by_section=group(lambda f: _section_of(f.pattern)),
        extras=[
            ExtraCount(pattern=pattern, count=count)
            for pattern, count in sorted(extra_counts.items(), key=lambda kv: -kv[1])
        ],
        uncertain=[f for f in fields if f.uncertain],
        cost=sum(r.cost for r in results),
    )
