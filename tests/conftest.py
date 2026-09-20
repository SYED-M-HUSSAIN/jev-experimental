"""Shared pytest wiring.

By default the suite runs offline: every request is answered from the recorded
responses in tests/fixtures, so anyone can clone the repo and see the numbers
without an API key.

    pytest                                  replay recorded responses (no key)
    JEV_MODE=record OPENROUTER_API_KEY=... pytest    call the API, update fixtures
    JEV_MODE=live   OPENROUTER_API_KEY=... pytest    call the API, record nothing
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jev_experiments.cassette import Cassette
from jev_experiments.client import JevClient

FIXTURES = Path(__file__).parent / "fixtures"
MODE = os.environ.get("JEV_MODE", "replay").strip().lower()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `live` tests in replay mode: they measure real run-to-run behaviour."""
    if MODE != "replay":
        return
    skip = pytest.mark.skip(reason="needs JEV_MODE=record or live (set OPENROUTER_API_KEY)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cassette() -> Cassette:
    return Cassette(FIXTURES, mode=MODE)


@pytest.fixture(scope="session")
def client(cassette: Cassette) -> JevClient:
    """One client for the whole session, so totals add up across experiments."""
    return JevClient(cassette=cassette)


@pytest.fixture(autouse=True)
def _rewind(cassette: Cassette):
    """Replay each test's recorded sequence from the start.

    Repeated identical requests replay in recorded order, so a test that asks
    the same question five times still sees the five real answers.
    """
    cassette.reset()
    yield


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    stats = Cassette(FIXTURES, mode=MODE).stats()
    terminalreporter.write_line(
        f"\njev: mode={MODE} · cassette files={stats['files']} · recorded responses={stats['responses']}"
    )
