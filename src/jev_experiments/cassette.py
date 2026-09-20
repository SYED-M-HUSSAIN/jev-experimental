"""Recorded HTTP responses, so the test suite runs offline by default.

Three modes, selected by the `JEV_MODE` environment variable (or explicitly):

  replay  never touch the network; serve the responses stored on disk (default)
  record  call the real API and append every response to the cassette
  live    call the real API and store nothing

A cassette is a directory of JSON files, one per distinct request payload,
named `<key>.json` where the key is a short sha256 of the canonical payload.

Each file holds a *sequence* of responses rather than a single one. Jev is a
probabilistic model: asking the identical question five times gives five
slightly different answers, and the consistency experiment depends on that
spread. Storing one response per key would flatten it into a constant and make
the offline suite measure something the live model never does. So replay walks
the recorded sequence in order; once it runs out it keeps handing back the last
entry (repeating is better than crashing a suite that grew an extra call).

The files are committed to git, so they are written deterministically: indented,
key-sorted, UTF-8, with a trailing newline.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

__all__ = ["Cassette", "CassetteMissError", "Mode", "MODES"]

Mode = Literal["replay", "record", "live"]
MODES: tuple[str, ...] = ("replay", "record", "live")

KEY_LENGTH = 16
PREVIEW_LENGTH = 200


class CassetteMissError(RuntimeError):
    """Replay was asked for a request that was never recorded."""

    def __init__(self, key: str, preview: str, directory: Path) -> None:
        self.key = key
        self.preview = preview
        self.directory = directory
        super().__init__(
            f"No cassette for key {key!r} in {directory}.\n"
            f"  request: {preview}\n"
            "  Re-record it with:  OPENROUTER_API_KEY=... JEV_MODE=record pytest ...\n"
            "  (an OpenRouter API key is required to record; replay needs no network)"
        )


def _canonical(payload: Any) -> str:
    """Byte-for-byte stable rendering of a payload, identical on every machine."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(payload: Any) -> str:
    """Stable short key for a request payload."""
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return digest[:KEY_LENGTH]


class Cassette:
    """A directory of recorded request/response sequences."""

    def __init__(self, directory: Path | str, mode: str | None = None) -> None:
        resolved = mode if mode is not None else os.environ.get("JEV_MODE", "replay")
        resolved = resolved.strip().lower()
        if resolved not in MODES:
            raise ValueError(f"invalid cassette mode {resolved!r}; expected one of {', '.join(MODES)}")
        self.mode: Mode = resolved  # type: ignore[assignment]
        self.directory = Path(directory)
        # Replay cursors live in memory only: each Cassette instance (i.e. each
        # test session) starts at the beginning of every recorded sequence.
        self._positions: dict[str, int] = {}

    # -- paths -------------------------------------------------------------
    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def key_for(self, payload: dict[str, Any]) -> str:
        return request_key(payload)

    # -- reading -----------------------------------------------------------
    def _load(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"malformed cassette file {path}: expected a JSON object")
        return data

    def replay(self, payload: dict[str, Any]) -> tuple[int, Any]:
        """Return the next recorded (status, body) for this payload."""
        key = request_key(payload)
        data = self._load(key)
        responses = data.get("responses") if data else None
        if not responses:
            preview = _canonical(payload)[:PREVIEW_LENGTH]
            raise CassetteMissError(key, preview, self.directory)

        index = self._positions.get(key, 0)
        # Past the end of the sequence we repeat the final response instead of
        # wrapping, so a longer run degrades into "the last answer" rather than
        # silently replaying the first one again.
        entry = responses[min(index, len(responses) - 1)]
        self._positions[key] = index + 1
        return int(entry["status"]), entry.get("body")

    # -- writing -----------------------------------------------------------
    def record(self, payload: dict[str, Any], status: int, body: Any) -> None:
        """Append one response to this payload's recorded sequence."""
        key = request_key(payload)
        data = self._load(key) or {"request": payload, "responses": []}
        responses = list(data.get("responses") or [])
        responses.append({"status": int(status), "body": body})
        self._write(key, {"request": payload, "responses": responses})

    def _write(self, key: str, data: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        # sort_keys keeps diffs stable; the responses list keeps its order,
        # which is the whole point of the cassette.
        text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8")

    # -- housekeeping ------------------------------------------------------
    def reset(self) -> None:
        """Rewind every sequence to its first recorded response."""
        self._positions.clear()

    def clear(self, key: str | None = None) -> None:
        """Delete cassette files: one key, or all of them."""
        if key is None:
            for path in self.directory.glob("*.json"):
                path.unlink()
            self._positions.clear()
            return
        self.path_for(key).unlink(missing_ok=True)
        self._positions.pop(key, None)

    def stats(self) -> dict[str, Any]:
        """Counts for a one-line summary at the end of a test run."""
        files = sorted(self.directory.glob("*.json")) if self.directory.is_dir() else []
        total = 0
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if isinstance(data, dict):
                total += len(data.get("responses") or [])
        return {
            "mode": self.mode,
            "directory": str(self.directory),
            "files": len(files),
            "responses": total,
            "replayed_keys": len(self._positions),
            "replays": sum(self._positions.values()),
        }

    def __repr__(self) -> str:
        return f"Cassette(directory={str(self.directory)!r}, mode={self.mode!r})"
