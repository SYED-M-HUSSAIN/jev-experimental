"""Out-of-vocabulary enum predictions: values outside the allowed set that may or may not map onto it."""

from __future__ import annotations

from pydantic import BaseModel

# Allowed values per field. "category" carries descriptions because the labels alone
# are not self-explanatory; "department" and "status" are bare lists, which is the
# common case and gives the judge less to work with.
ENUM_DEFS: dict[str, dict[str, str] | list[str]] = {
    "category": {
        "Electronics": "Powered hardware and accessories for it: computers, peripherals, displays, cables.",
        "Software": "Licences and subscriptions for programs and online services.",
        "Furniture": "Desks, chairs, storage and other physical office fittings.",
        "Services": "Labour billed by people: consulting, installation, training, support contracts.",
    },
    "department": ["Engineering", "Sales", "Finance", "HR", "Operations"],
    "status": ["pending", "approved", "rejected", "cancelled"],
}


class EnumCase(BaseModel):
    """A predicted value that is NOT in the field's allowed set, and whether it maps to the truth.

    `expected` is None when the mapping is genuinely arguable — those cases exist to
    check that a judge abstains or reports low confidence rather than guessing.
    """

    field: str
    ground_truth: str
    prediction: str
    expected: bool | None


ENUM_CASES: list[EnumCase] = [
    # --- category: should map to the correct allowed value -------------------
    EnumCase(field="category", ground_truth="Electronics", prediction="Computer peripherals", expected=True),
    EnumCase(field="category", ground_truth="Software", prediction="SaaS subscription", expected=True),
    EnumCase(field="category", ground_truth="Furniture", prediction="Office seating", expected=True),
    EnumCase(field="category", ground_truth="Services", prediction="Onsite installation labour", expected=True),
    EnumCase(field="category", ground_truth="Electronics", prediction="27-inch display", expected=True),
    # --- category: maps to a DIFFERENT allowed value, so it is wrong ---------
    # "Office chairs" is a clean Furniture match, so it cannot be excused as a loose
    # rendering of Electronics.
    EnumCase(field="category", ground_truth="Electronics", prediction="Office chairs", expected=False),
    EnumCase(field="category", ground_truth="Services", prediction="Annual licence renewal", expected=False),
    EnumCase(field="category", ground_truth="Software", prediction="Laptop docking station", expected=False),
    # --- category: fits nothing in the vocabulary ---------------------------
    # Pens and notebooks are none of the four allowed values; the right behaviour is
    # to fail rather than to force it into the nearest bucket.
    EnumCase(field="category", ground_truth="Electronics", prediction="Stationery", expected=False),
    # AMBIGUOUS: a managed cloud backup is sold as a subscription (Software) but is
    # substantially someone else running it for you (Services).
    EnumCase(field="category", ground_truth="Software", prediction="Managed cloud backup", expected=None),
    # AMBIGUOUS: a standing desk with a motor sits between Furniture and Electronics.
    EnumCase(field="category", ground_truth="Furniture", prediction="Electric sit-stand desk", expected=None),
    # --- department ----------------------------------------------------------
    EnumCase(field="department", ground_truth="Engineering", prediction="Platform team", expected=True),
    EnumCase(field="department", ground_truth="Finance", prediction="Accounts payable", expected=True),
    EnumCase(field="department", ground_truth="Sales", prediction="Account executives", expected=True),
    EnumCase(field="department", ground_truth="Operations", prediction="Logistics", expected=True),
    # Recruiting belongs to HR, not Operations — a confident mapping to the wrong member.
    EnumCase(field="department", ground_truth="Operations", prediction="Recruiting", expected=False),
    # AMBIGUOUS: payroll is run by Finance in some companies and HR in others, and the
    # bare list gives no descriptions to break the tie.
    EnumCase(field="department", ground_truth="Finance", prediction="Payroll", expected=None),
    # --- status --------------------------------------------------------------
    EnumCase(field="status", ground_truth="cancelled", prediction="withdrawn by customer", expected=True),
    EnumCase(field="status", ground_truth="approved", prediction="signed off", expected=True),
    EnumCase(field="status", ground_truth="pending", prediction="awaiting review", expected=True),
    EnumCase(field="status", ground_truth="rejected", prediction="declined", expected=True),
    # "not approved" reads as rejected, which is a different allowed value; it is not
    # a loose way of saying approved.
    EnumCase(field="status", ground_truth="approved", prediction="not approved", expected=False),
    EnumCase(field="status", ground_truth="pending", prediction="closed", expected=False),
    # AMBIGUOUS: "on hold" could be pending (still open, waiting) or cancelled
    # (stopped). Both readings are defensible without more context.
    EnumCase(field="status", ground_truth="pending", prediction="on hold", expected=None),
    # AMBIGUOUS: "expired" is neither cancelled by anyone nor still pending; the
    # vocabulary simply has no home for it.
    EnumCase(field="status", ground_truth="cancelled", prediction="expired", expected=None),
]
