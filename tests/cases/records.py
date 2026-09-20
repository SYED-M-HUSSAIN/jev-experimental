"""Ground-truth vs predicted record pairs, with per-field-path expectations for the field evaluator."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecordCase(BaseModel):
    """A ground-truth record, a model's extraction of it, and the verdict per field path.

    Keys of `expected_correct` are field PATHS into `ground_truth`: dotted for object
    members, bracketed for list positions (e.g. "customer.address.city", "items[1].qty").
    A path may also name a container ("items"), meaning "the list as a whole is correct",
    which catches missing and hallucinated entries.
    """

    name: str
    ground_truth: dict
    prediction: dict
    expected_correct: dict[str, bool]
    # Hard constraints the evaluator must honour for specific paths (e.g. exact match).
    rules: dict[str, str] = Field(default_factory=dict)
    # Soft guidance: what counts as a cosmetic difference for this field.
    hints: dict[str, str] = Field(default_factory=dict)
    # How to align list entries between the two records, keyed by list path.
    match_by: dict[str, str] = Field(default_factory=dict)


RECORD_PAIRS: list[RecordCase] = [
    RecordCase(
        name="invoice_formatting_noise",
        ground_truth={
            "invoice_id": "INV-2026-00841",
            "issue_date": "2026-10-03",
            "currency": "USD",
            "customer": {
                "name": "Acme Corporation",
                "tax_id": "US 47-1234567",
            },
            "subtotal": 1250,
            "tax_rate": 0.2,
            "total": 1500,
            "line_items": [
                {"description": "Annual platform licence", "amount": 1250},
            ],
        },
        prediction={
            "invoice_id": "INV-2026-0841",
            "issue_date": "October 3rd, 2026",
            "currency": "$",
            "customer": {
                "name": "ACME Corp.",
                "tax_id": "47-1234567",
            },
            "subtotal": "$1,250.00",
            "tax_rate": "17.5%",
            "total": "$1,500.00",
            "line_items": [
                {"description": "Annual platform license", "amount": "$1,250.00"},
            ],
        },
        expected_correct={
            # A dropped zero in an identifier is a different invoice, and this field is
            # under an exact-match rule, so no normalisation can rescue it.
            "invoice_id": False,
            # Same date, different rendering.
            "issue_date": True,
            # "$" is a symbol for the currency USD names; same value, looser notation.
            "currency": True,
            # Legal-suffix and casing differences only.
            "customer.name": True,
            # The country prefix is dropped but the identifying digits are unchanged.
            "customer.tax_id": True,
            "subtotal": True,
            # A genuine value error: 17.5% is not 20%. This is the case's real mistake,
            # separate from the ID typo.
            "tax_rate": False,
            "total": True,
            # British vs American spelling of the same word.
            "line_items[0].description": True,
            "line_items[0].amount": True,
        },
        rules={
            "invoice_id": "Must match character for character. Do not normalise, pad or re-hyphenate.",
        },
        hints={
            "customer.name": "Casing and legal-entity suffixes (Ltd/Inc/Corp.) are cosmetic.",
            "subtotal": "Numeric fields may arrive as formatted currency strings.",
            # Without saying so, dropping the country prefix reads as a changed ID.
            "customer.tax_id": "A leading country code may be omitted; the number itself must match.",
            "total": "Numeric fields may arrive as formatted currency strings.",
            "tax_rate": "A percentage string and a decimal fraction are the same value.",
        },
    ),
    RecordCase(
        name="purchase_order_nested_and_reordered",
        ground_truth={
            "po_number": "PO-4417",
            "customer": {
                "name": "Blue Harbor Supply Co.",
                "contact": {
                    "name": "J. Ramirez",
                    "email": "j.ramirez@blueharbor.example",
                    "phone": "+1-555-0142",
                },
                "billing": {
                    "address": {
                        "line1": "88 Quay Street",
                        "city": "Portland",
                        "postcode": "97204",
                        "country": "US",
                    },
                },
            },
            "shipping": {
                "method": "ground",
                "incoterms": "DAP",
                "window": {
                    "earliest": "2026-11-09",
                    "latest": "2026-11-16",
                },
            },
            "items": [
                {"sku": "KB-104", "description": "Mechanical keyboard, tenkeyless", "qty": 12, "unit_price": 79.0},
                {"sku": "MN-270", "description": "27-inch monitor, 1440p", "qty": 4, "unit_price": 245.0},
                {"sku": "DK-010", "description": "Docking station, dual USB-C", "qty": 6, "unit_price": 132.5},
                {"sku": "CB-221", "description": "USB-C cable, 2 m", "qty": 30, "unit_price": 9.75},
            ],
            "tags": ["priority", "contract-pricing", "partial-ship-ok"],
            "note": (
                "Deliver to the loading dock at the rear of the building; the front entrance "
                "has no step-free access. Call the contact 30 minutes before arrival."
            ),
        },
        prediction={
            "po_number": "PO-4417",
            "customer": {
                "name": "Blue Harbor Supply Company",
                "contact": {
                    "name": "Ramirez, J.",
                    "email": "J.Ramirez@BlueHarbor.example",
                    "phone": "+1 555 0142",
                },
                "billing": {
                    "address": {
                        "line1": "88 Quay St.",
                        "city": "Portland",
                        "postcode": "97205",
                        "country": "USA",
                    },
                },
            },
            "shipping": {
                "method": "ground freight",
                "incoterms": "DAP (Delivered at Place)",
                "window": {
                    "earliest": "2026-11-10",
                    "latest": "2026-11-16",
                },
            },
            # Deliberately out of order: DK first, KB last. CB-221 is missing and
            # MP-500 was invented.
            "items": [
                {"sku": "DK-010", "description": "Dual USB-C docking station", "qty": 6, "unit_price": 132.5},
                {"sku": "MN-270", "description": "27\" monitor (1440p)", "qty": 5, "unit_price": 245.0},
                {"sku": "MP-500", "description": "Mouse pad, large", "qty": 10, "unit_price": 14.0},
                {"sku": "KB-104", "description": "Tenkeyless mechanical keyboard", "qty": 12, "unit_price": 79.0},
            ],
            "tags": ["partial-ship-ok", "priority", "contract-pricing"],
            "note": (
                "Use the rear loading dock — there is no step-free access at the front. "
                "Phone the contact half an hour before you arrive."
            ),
        },
        expected_correct={
            "po_number": True,
            # "Company" spelled out instead of "Co." — same entity.
            "customer.name": True,
            # Surname-first rendering of the same person.
            "customer.contact.name": True,
            # Email domains are case-insensitive and mailbox casing is conventionally
            # ignored here, so this is the same address.
            "customer.contact.email": True,
            # Separator style only.
            "customer.contact.phone": True,
            "customer.billing.address.line1": True,
            "customer.billing.address.city": True,
            # 97205 is a different postcode from 97204 — a real extraction error, not
            # a formatting one.
            "customer.billing.address.postcode": False,
            # ISO alpha-2 vs alpha-3 for the same country.
            "customer.billing.address.country": True,
            "shipping.method": True,
            # The expansion of the Incoterm, not a different term.
            "shipping.incoterms": True,
            # Off by one day: the earliest ship date is wrong.
            "shipping.window.earliest": False,
            "shipping.window.latest": True,
            # Matched by SKU rather than position, so reordering alone is harmless.
            "items[0].qty": True,
            "items[0].description": True,
            # MN-270: ground truth says 4, the model said 5.
            "items[1].qty": False,
            "items[1].description": True,
            "items[2].qty": True,
            "items[2].description": True,
            # CB-221 is absent from the prediction entirely, so its quantity cannot be
            # correct.
            "items[3].qty": False,
            "items[3].description": False,
            # The list as a whole fails twice over: one entry dropped, one invented
            # (MP-500 appears nowhere in the ground truth).
            "items": False,
            # A list of primitives whose order carries no meaning.
            "tags": True,
            # Reworded, but every fact survives: rear loading dock, no step-free access
            # at the front, and 30 minutes' notice by phone.
            "note": True,
        },
        rules={
            "po_number": "Exact match.",
            "items[].sku": "Exact match; SKUs are identifiers, not descriptions.",
        },
        hints={
            "note": "Free text. Judge on facts preserved, not on wording or ordering.",
            "tags": "Unordered set of labels.",
            "shipping.method": "Carrier-service wording may vary.",
        },
        match_by={"items": "sku"},
    ),
    RecordCase(
        name="contact_record_absent_and_debatable",
        ground_truth={
            "first_name": "Dana",
            "last_name": "Whitfield",
            "email": "dana.whitfield@example.net",
            "company": None,
            "job_title": "Head of Operations",
            "source": "webinar",
        },
        prediction={
            "first_name": "Dana",
            "last_name": "Whitfield",
            "email": "Dana.Whitfield@Example.net",
            "company": "",
            "job_title": "Operations Lead",
            "source": "Webinar signup",
        },
        expected_correct={
            "first_name": True,
            "last_name": True,
            "email": True,
            # DEBATABLE: null and empty string both encode "no company was given". A
            # stricter evaluator that distinguishes absent from empty would say False;
            # this suite treats them as the same fact.
            "company": True,
            # DEBATABLE, labelled False: "Head of Operations" and "Operations Lead" are
            # near-synonyms in casual use, but they are different job titles as written
            # and a CRM would not accept one as a transcription of the other.
            "job_title": False,
            # Same enum member with extra wording around it.
            "source": True,
        },
        hints={
            "email": "Compare case-insensitively.",
            "company": "An empty value and a null both mean the field was not present.",
            "source": "Acquisition channel: a longer label for the same channel is a match.",
        },
    ),
]
