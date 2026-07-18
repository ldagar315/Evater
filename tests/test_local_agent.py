from __future__ import annotations

import asyncio

import pytest

from backend.app.ai.local_agent import (
    LocalAgentConfigurationError,
    build_local_agent,
    build_local_session,
    run_local_turn,
)


def test_local_agent_persists_session_history(tmp_path):
    async def scenario():
        agent = build_local_agent("mock")
        session = build_local_session(
            session_id="contract-session",
            db_path=tmp_path / "agent.sqlite3",
        )

        first = await run_local_turn(agent, session, "Start a study session.")
        second = await run_local_turn(agent, session, "Continue the session.")
        items = await session.get_items()

        assert first == "Local Evater tutor ready. Tell me what you want to practise."
        assert second == "I remember this session. Let’s continue from your previous answer."
        assert len(items) == 4

    asyncio.run(scenario())


def test_live_provider_requires_a_local_key(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LocalAgentConfigurationError):
        build_local_agent("cerebras")

    with pytest.raises(LocalAgentConfigurationError):
        build_local_agent("openai")
