"""Short source documents plus claims labelled supported / contradicted / not_mentioned."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DOCS: dict[str, str] = {
    "product_spec": (
        "Meridian X2 Desktop Scanner — Technical Specification\n\n"
        "The Meridian X2 is a flatbed document scanner for small offices. It captures at up to "
        "1200 dpi optical resolution and scans a single-sided A4 page in 3.4 seconds in colour. "
        "The automatic document feeder holds 50 sheets and supports double-sided capture in one "
        "pass. Connection is over USB-C or 2.4 GHz Wi-Fi; there is no wired Ethernet port. "
        "The unit weighs 3.2 kg and measures 448 x 300 x 91 mm with the feeder tray closed. "
        "Power draw is 18 W while scanning and under 1 W in standby. Supplied software exports "
        "to PDF, PDF/A, TIFF and PNG. The scanner ships with a 3 year limited hardware warranty "
        "covering manufacturing defects; consumables such as the feeder roller are excluded."
    ),
    "policy_page": (
        "Refunds and Cancellations\n\n"
        "You may cancel a subscription at any time from the Billing page. Cancellation takes "
        "effect at the end of the current billing period, and the service remains fully available "
        "until then. We do not pro-rate partial months. New customers on an annual plan may "
        "request a full refund within 30 days of their first payment; after day 30, annual plans "
        "are non-refundable. Monthly plans are not eligible for refunds at any point, though we "
        "will review billing errors on request. Refunds are returned to the original payment "
        "method and typically appear within 5 to 10 business days. Usage-based overage charges "
        "are billed in arrears and cannot be refunded once the usage has been incurred."
    ),
    "invoice_text": (
        "Invoice INV-2026-00912\n\n"
        "Issued 14 October 2026, payable by 13 November 2026. Billed to Harbourline Studios, "
        "22 Fenwick Road, Leeds LS3 1QT, United Kingdom.\n\n"
        "Line 1: Growth plan, 8 seats, 1 October to 31 October 2026 — 8 x 45.00 = 360.00.\n"
        "Line 2: API overage, 120,000 calls above the included allowance — 96.00.\n"
        "Line 3: Onboarding session (2 hours) — 240.00.\n\n"
        "Subtotal 696.00. VAT at 20% — 139.20. Total due 835.20 GBP.\n"
        "Payment by bank transfer only; card payments are not accepted on this account. "
        "Late payment attracts interest at 2% per month on the outstanding balance."
    ),
}


class ClaimCase(BaseModel):
    """One claim checked against one document in DOCS."""

    doc: str
    claim: str
    verdict: Literal["supported", "contradicted", "not_mentioned"]


CLAIM_CASES: list[ClaimCase] = [
    # --- product_spec --------------------------------------------------------
    ClaimCase(
        doc="product_spec",
        claim="The document feeder takes 50 sheets.",
        verdict="supported",
    ),
    # Unit conversion: the doc says "3 year", the claim says "36 months". Same span.
    ClaimCase(
        doc="product_spec",
        claim="The hardware warranty lasts 36 months.",
        verdict="supported",
    ),
    # Subtly wrong number: the spec says 3.4 seconds, not 4.3.
    ClaimCase(
        doc="product_spec",
        claim="A colour A4 page scans in 4.3 seconds.",
        verdict="contradicted",
    ),
    # The doc states explicitly that there is no Ethernet port.
    ClaimCase(
        doc="product_spec",
        claim="The scanner can be connected to the network over Ethernet.",
        verdict="contradicted",
    ),
    # Price appears nowhere in a spec sheet.
    ClaimCase(
        doc="product_spec",
        claim="The Meridian X2 costs $399.",
        verdict="not_mentioned",
    ),
    # DEBATABLE, and a good example of why the boundary has to be stated. The spec
    # lists PDF, PDF/A, TIFF and PNG. Read as an exhaustive list, JPEG's absence
    # contradicts the claim; read as examples, the spec is simply silent. The model
    # takes the closed-world reading (0.71 contradicted / 0.29 not_mentioned), which
    # is the more useful one for a spec sheet, so that is the label here.
    ClaimCase(
        doc="product_spec",
        claim="Scans can be exported as JPEG.",
        verdict="contradicted",
    ),
    # --- policy_page ---------------------------------------------------------
    ClaimCase(
        doc="policy_page",
        claim="An annual subscriber who paid three weeks ago can still get a full refund.",
        verdict="supported",
    ),
    # The page says monthly plans are never refundable.
    ClaimCase(
        doc="policy_page",
        claim="Monthly subscribers can request a refund within 30 days.",
        verdict="contradicted",
    ),
    # Paraphrase of "cancellation takes effect at the end of the current billing period"
    # plus "we do not pro-rate partial months".
    ClaimCase(
        doc="policy_page",
        claim="Cancelling mid-month does not produce a partial refund; access continues to the period end.",
        verdict="supported",
    ),
    # Nothing on the page discusses who may cancel or what permissions are needed.
    ClaimCase(
        doc="policy_page",
        claim="Only the account owner may cancel the subscription.",
        verdict="not_mentioned",
    ),
    # --- invoice_text --------------------------------------------------------
    # Requires reading the arithmetic on the page: 8 x 45.00 = 360.00.
    ClaimCase(
        doc="invoice_text",
        claim="The seat line comes to 360.00 for eight seats at 45.00 each.",
        verdict="supported",
    ),
    # Subtly wrong number: the total is 835.20, not 853.20.
    ClaimCase(
        doc="invoice_text",
        claim="The total due is 853.20 GBP.",
        verdict="contradicted",
    ),
    ClaimCase(
        doc="invoice_text",
        claim="The invoice can be settled by credit card.",
        verdict="contradicted",
    ),
    # A 30-day gap between issue and due date is derivable (14 Oct to 13 Nov), so this
    # is entailed rather than merely absent.
    ClaimCase(
        doc="invoice_text",
        claim="Payment terms are 30 days from the issue date.",
        verdict="supported",
    ),
    # The invoice gives the overage count but never the size of the included allowance
    # it sits on top of, so the figure cannot be checked either way.
    ClaimCase(
        doc="invoice_text",
        claim="The Growth plan includes 500,000 API calls before overage begins.",
        verdict="not_mentioned",
    ),
]
