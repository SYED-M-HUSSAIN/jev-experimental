"""Text pairs that separate semantic equivalence from surface-string similarity."""

from __future__ import annotations

# Pairs that share almost no vocabulary but mean the same thing.
# A string/embedding-overlap baseline should fail most of these; a semantic
# judge should call every one of them equivalent.
SAME_MEANING: list[tuple[str, str]] = [
    # Double negative: "may with consent" and "may not without consent" state the same rule.
    ("The licence may be transferred with written consent", "The licence may not be transferred without written consent"),
    # support
    ("Customer wants their money back", "Refund requested"),
    ("The user can no longer sign in to the dashboard", "Login is broken for this account"),
    ("Please stop billing me every month", "Cancel my subscription"),
    # medical
    ("Hypertension", "High blood pressure"),
    ("Patient reports difficulty breathing on exertion", "Shortness of breath when active"),
    # legal / policy
    ("Either party may end this agreement with 30 days' notice", "This contract is terminable at will on a month's notice"),
    ("The vendor is not liable for indirect damages", "Consequential loss is excluded from the supplier's liability"),
    # logistics
    ("The parcel left our warehouse this morning", "Order has been dispatched"),
    ("Delivery is delayed due to weather", "Shipment will arrive later than promised because of a storm"),
    # finance
    ("Q3 2026", "July–September 2026"),
    ("Invoice remains unpaid past its due date", "The account is in arrears"),
    ("Revenue grew by a fifth year over year", "Annual revenue was up 20% compared with last year"),
]

# Pairs that look alike token-for-token but assert different things.
# These are the false positives a naive similarity metric produces.
DIFFERENT_MEANING: list[tuple[str, str]] = [
    # support
    ("Approved", "Not approved"),
    ("The refund has been issued", "The refund has been requested"),
    ("We were unable to reproduce the bug", "We were able to reproduce the bug"),
    # legal / policy
    ("Payment due before delivery", "Payment due after delivery"),
    # NOTE: near-identical wording, opposite obligation — "shall" binds, "may" does not.
    ("The supplier shall provide monthly reports", "The supplier may provide monthly reports"),
    # logistics
    ("Ships within 5 business days", "Ships within 5 days"),
    ("Delivered to the mailroom", "Delivered to the recipient"),
    # finance
    ("Renewal price increases by 10%", "Renewal price increases to 10%"),
    ("Discount applies to the first year only", "Discount applies from the first year onward"),
    ("Balance of $2,400 outstanding", "Balance of $2,400 paid"),
    # medical
    ("No history of diabetes", "History of diabetes"),
]
