"""Labelled yes/no and single-choice cases for measuring judge accuracy and calibration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class NoulCase(BaseModel):
    """One boolean question asked about one piece of state."""

    state: Any
    question: str
    expected: bool
    # `hard` marks cases where a shallow keyword reading gives the wrong answer;
    # useful for reporting accuracy split by difficulty.
    hard: bool = False


class ChoiceCase(BaseModel):
    """One single-label classification over a fixed option set."""

    state: Any
    expected: str
    hard: bool = False


_COMPLAINT_Q = "Does this message express a complaint about us?"

COMPLAINT_CASES: list[NoulCase] = [
    NoulCase(
        state="Third time this week the export has failed. This is getting ridiculous.",
        question=_COMPLAINT_Q,
        expected=True,
    ),
    NoulCase(
        state="Hi! Quick question — where do I find my invoices in the new UI?",
        question=_COMPLAINT_Q,
        expected=False,
    ),
    NoulCase(
        state="Just wanted to say the new scheduling view is excellent. Thanks.",
        question=_COMPLAINT_Q,
        expected=False,
    ),
    # HARD: explicit disclaimer of complaining, but the substance is still a grievance
    # about our billing. The disclaimer is politeness framing, not the content.
    NoulCase(
        state=(
            "Not complaining at all, just curious — why did my bill jump from $49 to $89 "
            "without any notice? Seems like a lot."
        ),
        question=_COMPLAINT_Q,
        expected=True,
        hard=True,
    ),
    # HARD: sarcasm. Every sentiment word is positive; the meaning is the opposite.
    NoulCase(
        state="Love that the app logs me out mid-form. Really great use of my afternoon.",
        question=_COMPLAINT_Q,
        expected=True,
        hard=True,
    ),
    # HARD: the complaint targets a THIRD party (their previous vendor). We are the
    # subject of praise, so the answer to "about us" is False.
    NoulCase(
        state="My old provider was terrible, which is why I switched to you. Night and day.",
        question=_COMPLAINT_Q,
        expected=False,
        hard=True,
    ),
    # HARD: extremely polite wrapping, but it is a complaint about our response times.
    NoulCase(
        state=(
            "I hope you don't mind me raising this, and I do appreciate how busy you must be, "
            "but I've now been waiting eleven days for a reply to ticket 4471."
        ),
        question=_COMPLAINT_Q,
        expected=True,
        hard=True,
    ),
    # HARD: negative words ("broken", "awful") describe the customer's own setup, not us.
    NoulCase(
        state="Our internal VPN is awful and keeps dropping — is there a way to allowlist your IPs?",
        question=_COMPLAINT_Q,
        expected=False,
        hard=True,
    ),
    NoulCase(
        state=(
            "Cancelled my plan yesterday and was still charged today. Please sort this out."
        ),
        question=_COMPLAINT_Q,
        expected=True,
    ),
    # HARD/BORDERLINE: a feature request phrased with mild frustration. Labelled False
    # because the substance is a request for something that does not exist yet, not a
    # grievance about a failure. Reasonable annotators could disagree here.
    NoulCase(
        state="Would be nice if reports could be exported to CSV. Copy-pasting is a pain.",
        question=_COMPLAINT_Q,
        expected=False,
        hard=True,
    ),
]


_GROUNDING_Q = "Is every claim in the answer supported by the supplied documents?"

GROUNDING_CASES: list[NoulCase] = [
    NoulCase(
        state={
            "docs": ["The Starter plan includes 3 seats and 10 GB of storage."],
            "answer": "Starter comes with 3 seats.",
        },
        question=_GROUNDING_Q,
        expected=True,
    ),
    NoulCase(
        state={
            "docs": ["The Starter plan includes 3 seats and 10 GB of storage."],
            "answer": "Starter comes with 5 seats.",
        },
        question=_GROUNDING_Q,
        expected=False,
    ),
    # HARD: requires arithmetic. $30/mo * 12 = $360, less 20% = $288. The answer is
    # entailed by the docs even though the number "288" appears nowhere in them.
    NoulCase(
        state={
            "docs": [
                "The Pro plan is billed at $30 per user per month.",
                "Customers who pay annually receive a 20% discount off the monthly rate.",
            ],
            "answer": "Paying annually for one Pro user costs $288 for the year.",
        },
        question=_GROUNDING_Q,
        expected=True,
        hard=True,
    ),
    # HARD: same arithmetic setup, but the discount was applied to the monthly figure
    # only ($30 * 12 = $360, answer says $345). Unsupported.
    NoulCase(
        state={
            "docs": [
                "The Pro plan is billed at $30 per user per month.",
                "Customers who pay annually receive a 20% discount off the monthly rate.",
            ],
            "answer": "Paying annually for one Pro user costs $345 for the year.",
        },
        question=_GROUNDING_Q,
        expected=False,
        hard=True,
    ),
    # HARD: the added claim ("SOC 2 certified") is plausible and may well be true of a
    # real SaaS, but nothing in the docs supports it. Grounding is about the docs, not
    # about the world.
    NoulCase(
        state={
            "docs": [
                "All data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
            ],
            "answer": "Data is encrypted at rest with AES-256, and the service is SOC 2 certified.",
        },
        question=_GROUNDING_Q,
        expected=False,
        hard=True,
    ),
    # HARD: paraphrase only. "Within two business days" == "no later than 48 working hours"
    # for this purpose; no new content is introduced.
    NoulCase(
        state={
            "docs": ["Support tickets on the Business plan are answered within two business days."],
            "answer": "Business-plan tickets get a reply no later than two working days after they are opened.",
        },
        question=_GROUNDING_Q,
        expected=True,
        hard=True,
    ),
    NoulCase(
        state={
            "docs": [
                "Refunds are available within 30 days of purchase.",
                "Refunds are not available on usage-based overage charges.",
            ],
            "answer": "You can get a refund within 30 days, including on overage charges.",
        },
        question=_GROUNDING_Q,
        expected=False,
    ),
    # HARD: hedged answer. The docs are silent on mobile apps and the answer says so
    # explicitly rather than asserting anything — so nothing unsupported is claimed.
    NoulCase(
        state={
            "docs": ["The web application supports Chrome, Firefox and Safari."],
            "answer": "The docs list Chrome, Firefox and Safari; they do not say whether a mobile app exists.",
        },
        question=_GROUNDING_Q,
        expected=True,
        hard=True,
    ),
]


TEAMS: dict[str, str] = {
    "billing": "Invoices, payment methods, refunds, plan pricing, tax and dunning.",
    "technical": "Bugs, errors, outages, integrations, API problems and anything that is broken.",
    "account": "Logins, passwords, SSO, seats, permissions and closing or merging accounts.",
    "sales": "Pre-purchase questions, quotes, plan comparisons, upgrades and contract terms.",
}

ROUTING_CASES: list[ChoiceCase] = [
    ChoiceCase(
        state="I was charged twice for the September invoice. Can you refund one?",
        expected="billing",
    ),
    ChoiceCase(
        state="The API returns 500 on every POST to /v2/events since this morning.",
        expected="technical",
    ),
    ChoiceCase(
        state="Can you remove a colleague who has left and free up their seat?",
        expected="account",
    ),
    ChoiceCase(
        state="We're 40 people and evaluating you against two other tools. Can we get a quote?",
        expected="sales",
    ),
    # HARD: the surface topic is billing (it happens on the billing page), but the
    # problem is a front-end defect, which technical owns.
    ChoiceCase(
        state="Getting 'Uncaught TypeError: cannot read properties of null' in the console on the billing page — it won't load.",
        expected="technical",
        hard=True,
    ),
    # HARD: mentions "password" but the actual ask is about upgrading, i.e. sales.
    ChoiceCase(
        state="I reset my password fine, thanks. What I actually want to know is what the Enterprise tier adds over Pro.",
        expected="sales",
        hard=True,
    ),
    # HARD: says "invoice", but the failure is that SSO blocks login, so nobody can
    # reach the invoice. The blocking issue is the account/SSO one.
    ChoiceCase(
        state="I can't download my invoice because our Okta SSO login loops back to the sign-in screen.",
        expected="account",
        hard=True,
    ),
    # HARD/BORDERLINE: "upgrade my plan" is self-serve in-app, so this could be billing;
    # labelled sales because the ask is about which plan to move to, not about payment.
    # A reasonable ops team might route this either way.
    ChoiceCase(
        state="We keep hitting the 10k-row export cap. Which plan should we move to?",
        expected="sales",
        hard=True,
    ),
]
