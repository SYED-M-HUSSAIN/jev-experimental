"""Cases where the same question has a different correct answer once the context changes."""

from __future__ import annotations

# One refund request, held constant, evaluated against different customer history and
# policy. The question in every row is: should this refund be auto-approved without a
# human reviewing it? Nothing about the request itself decides the answer.
_REFUND_REQUEST = "I bought the annual plan last week and it isn't what I expected. I'd like a refund."

REFUND_CONTEXTS: list[tuple[str, dict, bool]] = [
    (
        "inside_window_clean_history",
        {
            "request": _REFUND_REQUEST,
            "policy": {
                "refund_window_days": 30,
                "auto_approve_limit_usd": 500,
                "auto_approve_requires_clean_history": True,
            },
            "customer": {
                "days_since_purchase": 7,
                "amount_usd": 288,
                "prior_refunds_12mo": 0,
                "account_flags": [],
            },
        },
        True,
    ),
    (
        "outside_refund_window",
        {
            "request": _REFUND_REQUEST,
            "policy": {
                "refund_window_days": 30,
                "auto_approve_limit_usd": 500,
                "auto_approve_requires_clean_history": True,
            },
            # Same request, same amount — but 62 days have passed, so the window has closed.
            "customer": {
                "days_since_purchase": 62,
                "amount_usd": 288,
                "prior_refunds_12mo": 0,
                "account_flags": [],
            },
        },
        False,
    ),
    (
        "over_auto_approve_limit",
        {
            "request": _REFUND_REQUEST,
            "policy": {
                "refund_window_days": 30,
                "auto_approve_limit_usd": 500,
                "auto_approve_requires_clean_history": True,
            },
            # In-window, but the amount exceeds the auto-approval ceiling.
            "customer": {
                "days_since_purchase": 7,
                "amount_usd": 2880,
                "prior_refunds_12mo": 0,
                "account_flags": [],
            },
        },
        False,
    ),
    (
        "repeat_refunder",
        {
            "request": _REFUND_REQUEST,
            "policy": {
                "refund_window_days": 30,
                "auto_approve_limit_usd": 500,
                "auto_approve_requires_clean_history": True,
            },
            # Three refunds in a year is not a clean history under this policy.
            "customer": {
                "days_since_purchase": 5,
                "amount_usd": 288,
                "prior_refunds_12mo": 3,
                "account_flags": [],
            },
        },
        False,
    ),
    (
        "generous_policy_same_late_request",
        {
            "request": _REFUND_REQUEST,
            # Identical late request to `outside_refund_window`, but this tenant runs a
            # 90-day window — so the correct answer flips purely on policy.
            "policy": {
                "refund_window_days": 90,
                "auto_approve_limit_usd": 5000,
                "auto_approve_requires_clean_history": False,
            },
            "customer": {
                "days_since_purchase": 62,
                "amount_usd": 288,
                "prior_refunds_12mo": 0,
                "account_flags": [],
            },
        },
        True,
    ),
    (
        "fraud_flag_overrides_everything",
        {
            "request": _REFUND_REQUEST,
            "policy": {
                "refund_window_days": 90,
                "auto_approve_limit_usd": 5000,
                "auto_approve_requires_clean_history": False,
                "never_auto_approve_flags": ["chargeback_open", "suspected_fraud"],
            },
            # Every numeric condition passes; a single flag still forces human review.
            "customer": {
                "days_since_purchase": 3,
                "amount_usd": 288,
                "prior_refunds_12mo": 0,
                "account_flags": ["chargeback_open"],
            },
        },
        False,
    ),
]


INTENTS: dict[str, str] = {
    "upgrade": "The customer is agreeing to move to a larger or more expensive plan.",
    "cancel": "The customer is agreeing to end their subscription or let it lapse.",
    "refund": "The customer is agreeing to be refunded for a charge already taken.",
}

# The latest message is the same three words every time. Only the preceding turn
# determines what the customer has just consented to.
_LATEST = "yes, go ahead"

CONVERSATION_CONTEXTS: list[tuple[str, dict, str]] = [
    (
        "agent_offered_upgrade",
        {
            "history": [
                {"role": "customer", "text": "We keep hitting the seat limit on Starter."},
                {"role": "agent", "text": "I can move you to Pro for $30 per seat per month, effective today. Shall I?"},
            ],
            "latest": _LATEST,
        },
        "upgrade",
    ),
    (
        "agent_offered_cancellation",
        {
            "history": [
                {"role": "customer", "text": "We're not using this any more."},
                {"role": "agent", "text": "I can cancel the subscription at the end of your current term. Would you like me to?"},
            ],
            "latest": _LATEST,
        },
        "cancel",
    ),
    (
        "agent_offered_refund",
        {
            "history": [
                {"role": "customer", "text": "I was billed on the 3rd even though I'd already downgraded."},
                {"role": "agent", "text": "You're right, that charge shouldn't have gone through. I can refund the $89 to your card."},
            ],
            "latest": _LATEST,
        },
        "refund",
    ),
    (
        "two_offers_last_one_wins",
        {
            # The agent mentioned cancelling earlier, then landed on a refund. "Yes"
            # attaches to the most recent open offer, not the first one raised.
            "history": [
                {"role": "customer", "text": "I want to cancel and get my money back."},
                {"role": "agent", "text": "Before cancelling — would a refund of this month's charge help, keeping the account open?"},
                {"role": "customer", "text": "Maybe. What would that look like?"},
                {"role": "agent", "text": "I'd return the $89 charged on the 3rd and leave your plan running."},
            ],
            "latest": _LATEST,
        },
        "refund",
    ),
    (
        "upgrade_framed_as_cancellation_save",
        {
            # Heavy "cancel" vocabulary in the history, but the standing offer the
            # customer is agreeing to is an upgrade.
            "history": [
                {"role": "customer", "text": "Please cancel my account, it's too limited."},
                {"role": "agent", "text": "Sorry to hear it. Before I cancel — the Pro tier removes that limit and I can apply a 30% discount for six months. Want me to switch you instead?"},
            ],
            "latest": _LATEST,
        },
        "upgrade",
    ),
    (
        "refund_mentioned_but_offer_is_cancellation",
        {
            "history": [
                {"role": "customer", "text": "Can I get a refund for the last three months?"},
                {"role": "agent", "text": "Refunds only cover the current billing period, so I can't return those three months. What I can do is cancel the renewal so you're not charged again — shall I do that?"},
            ],
            "latest": _LATEST,
        },
        "cancel",
    ),
]
