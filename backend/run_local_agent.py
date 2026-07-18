"""Run the local Evater Agents SDK prototype.

Examples:
  python3 backend/run_local_agent.py --provider mock
  CEREBRAS_API_KEY=... python3 backend/run_local_agent.py --provider cerebras
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from app.ai.local_agent import build_local_agent, build_local_session, run_local_turn


async def main(provider: str, db_path: str | None) -> None:
    session_path = db_path or tempfile.NamedTemporaryFile(
        prefix="evater-agent-", suffix=".sqlite3", delete=False
    ).name
    agent = build_local_agent(provider)
    session = build_local_session(db_path=session_path)

    first = await run_local_turn(agent, session, "Start a short study session.")
    second = await run_local_turn(agent, session, "Continue from what we discussed.")

    print(f"session_db={Path(session_path)}")
    print(f"turn_1={first}")
    print(f"turn_2={second}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("mock", "cerebras", "openai"),
        default=None,
        help="Provider to use; defaults to EVATER_LOCAL_AGENT_PROVIDER or mock.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite path for durable session history; defaults to a temporary file.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.provider, args.db_path))
