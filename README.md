# Jev Experiments

Jev, from [TypeSafe AI](https://typesafe.ai), is described everywhere as a fast, cheap
alternative to an LLM for structured output. That description is half right, and the half
that is wrong matters: **it cannot produce structured output at all.** It cannot return a
name, an amount or a date.

This repo is what it can do, as a test suite you can run.

## Quick start

```bash
git clone <this repo> && cd Jev-Experiments
pip install -e ".[dev]"
pytest -s
```

That's it — **no API key needed**. Every API response is recorded in `tests/fixtures`, so
the suite replays them offline in under a second and prints the same numbers I got:

```
capabilities   extraction request rejected: "Expected 'noul' | 'choice' | 'score'"
meaning        24/24 pairs judged correctly (12 paraphrases accepted, 12 near-misses rejected)
accuracy       yes/no 100% (n=18) · routing 8/8 · Brier 0.038 (0 = perfect, 0.25 = coin flip)
consistency    verdict unchanged across repeats, and when options are reordered or renamed
context        the same question flips with the policy, the history and the customer record
grounding      15/15 claims classified supported / contradicted / not mentioned
field_eval     enum mapping 20/20 · formatting-only differences pass, changed values fail
```

To call the real API instead:

```bash
cp .env.example .env               # add an OpenRouter key
export OPENROUTER_API_KEY=...
JEV_MODE=record pytest             # call the API and update the fixtures
JEV_MODE=live   pytest             # call the API, record nothing
```

## What Jev is

You send a **state** (a string or any nested JSON) and **typed questions** about it. You get
back probabilities, never prose. There are exactly three question types:

| Type | Question | Answer |
|---|---|---|
| `noul` | yes/no | probability that the statement is true |
| `choice` | pick one of your options | a probability per option, plus confidence |
| `score` | position on an ordered scale you define | a continuous score, plus the distribution |

Questions in one request are evaluated in parallel, so asking ten costs barely more time
than asking one.

```python
from jev_experiments import JevClient

client = JevClient()
answer = client.ask(
    "The deploy failed twice and customers are seeing 500s. Can someone look now?",
    "Does this need attention right now?",
)
answer.noul        # 0.98
answer.uncertain   # False -> confident enough to act on automatically
```

## What it cannot do

- **Extraction.** No names, amounts, dates or IDs. `test_capabilities.py` shows the API
  rejecting the attempt. You still need an LLM with structured output for that.
- **Generation.** No summaries, replies or explanations.
- **Open questions.** You have to know the possible answers in advance.
- **Anything outside the state.** No tools, no retrieval, no search: it sees only what you send.

## What it is good for

- **Verification** — is this answer supported by the retrieved documents, is this extracted
  value actually in the source? The strongest result in this repo.
- **Routing and triage** — which team, which queue, which priority.
- **Guardrails** — is this tool call risky, does this text contain personal data.
- **Model routing** — is this request simple enough for a cheap model.
- **Scoring at volume** — rate every production response instead of a sample.

## The experiments

| File | Question it answers |
|---|---|
| `test_capabilities.py` | What can you actually ask? Includes the API **rejecting an extraction request**. |
| `test_meaning.py` | Does it compare meaning or wording? "Approved" vs "Not approved" is the test. |
| `test_accuracy.py` | How often is it right, and are its probabilities calibrated? |
| `test_consistency.py` | Same question twice: same answer? Does option order or naming change it? |
| `test_context.py` | Does the answer track the context you pass, or just the sentence? |
| `test_field_eval.py` | Ground truth vs LLM output, field by field: nested objects, lists, out-of-list enums. |
| `test_grounding.py` | Is this claim supported by the document, contradicted by it, or absent from it? |

Test cases live in `tests/cases/` as plain data — **inputs with their expected answers** —
so the tests measure accuracy instead of printing whatever came back. All of it is
invented: generic SaaS, e-commerce and support-desk material, no real company data.

Two results worth reading the code for:

- **`test_accuracy.py` sat at 83% until the criteria were written properly.** With
  `true: "Yes" / false: "No"`, a customer complaining about *their own VPN* counts as a
  complaint about you, and a feature request counts as a grievance. Spelling out what each
  answer means took it to 100%. The meaning has to live in the criteria, not the question name.
- **`tests/cases/grounding.py` contains a claim the model and I disagreed on.** A spec lists
  PDF, PDF/A, TIFF and PNG; is "exports as JPEG" contradicted, or simply not mentioned? The
  model took the closed-world reading at 0.71. The comment there keeps the disagreement
  visible rather than hiding it.

While writing the cases I also mislabelled a pair as opposites — "may be transferred **with**
written consent" vs "may **not** be transferred **without** written consent" — which are
logically the same rule. The model was right and the label was wrong.

## Using the probabilities

The answer is a number, not a label, and that is the part worth designing around: automate
above a high threshold, send the middle band to a person. TypeSafe's own guidance is to
validate those thresholds on your own domain rather than assuming defaults transfer, and
`test_accuracy.py` is the shape of that check — accuracy split by easy/hard, a Brier score,
and confidence buckets comparing claimed confidence against how often it was actually right.

## Layout

```
src/jev_experiments/
  client.py      typed client: noul / choice / score, errors, Brier score
  cassette.py    record & replay, so the suite runs offline
  field_eval.py  ground truth vs prediction: nested objects, list pairing,
                 out-of-vocabulary enum mapping, dataset-level report
tests/
  cases/         the test data: inputs plus expected answers
  fixtures/      recorded API responses (committed, ~660 KB)
```

The API is reached through OpenRouter's Decisions endpoint, model `~typesafe/jev-latest`.
Recorded with Jev 1.13 in September 2026; re-record with `JEV_MODE=record` to check whether
the numbers still hold.
