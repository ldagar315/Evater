"""Tests for the transient Supabase transport retry helper.

Production evidence (Oct 24, Modal logs): httpx.RemoteProtocolError
"Server disconnected" killed block submits mid-request even though the
immediately-following retry succeeded.
"""
import httpx
import pytest

from app.supabase_client import run_with_retry


def test_returns_first_success_without_retry():
    calls = []
    result = run_with_retry(lambda: calls.append(1) or "ok", base_delay=0)
    assert result == "ok"
    assert len(calls) == 1


def test_retries_after_transient_disconnect_then_succeeds():
    state = {"calls": 0}

    def flaky():
        state["calls"] += 1
        if state["calls"] < 3:
            raise httpx.RemoteProtocolError("Server disconnected")
        return "recovered"

    assert run_with_retry(flaky, base_delay=0) == "recovered"
    assert state["calls"] == 3


def test_gives_up_after_attempts_exhausted():
    state = {"calls": 0}

    def always_down():
        state["calls"] += 1
        raise httpx.RemoteProtocolError("Server disconnected")

    with pytest.raises(httpx.RemoteProtocolError):
        run_with_retry(always_down, attempts=3, base_delay=0)
    assert state["calls"] == 3


def test_non_transient_errors_are_not_retried():
    state = {"calls": 0}

    def bad():
        state["calls"] += 1
        raise ValueError("not a transport problem")

    with pytest.raises(ValueError):
        run_with_retry(bad, base_delay=0)
    assert state["calls"] == 1
