"""Checkpoint 3 regression tests using local model responses."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from google.genai import types

from agents.agent import create_blue_agent
from assignment.pipeline import build_production_plugins, is_egress_allowed
from assignment.rate_limiter import RateLimitPlugin
from core.openai_runtime import OpenAIRunner
from core.utils import chat_with_agent


APPROVED_URL = "https://api.vinbank.example/v1/transfers"


def test_blue_rate_limit_tracks_user_and_stops_before_model(monkeypatch):
    create = Mock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Account statement is ready."))]
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(OpenAIRunner, "_client", lambda self: client)
    agent, runner = create_blue_agent(build_production_plugins(max_requests=1))

    async def run():
        query = "What is my account balance?"
        first, _ = await chat_with_agent(agent, runner, query)
        limited, _ = await chat_with_agent(agent, runner, query)

        assert first == "Account statement is ready."
        assert limited.startswith("Rate limit exceeded.")
        assert create.call_count == 1

        # Verify per-user sliding window isolation directly on RateLimitPlugin
        rl = RateLimitPlugin(max_requests=1, window_seconds=60)
        msg = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        assert await rl.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id="first-user"), user_message=msg
        ) is None
        blocked = await rl.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id="first-user"), user_message=msg
        )
        assert blocked is not None and blocked.parts[0].text.startswith("Rate limit exceeded.")
        assert await rl.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id="second-user"), user_message=msg
        ) is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "destination",
    [
        "https://[",
        "https://api.vinbank.example:invalid/v1/transfers",
        "https://api.vinbank.example:65536/v1/transfers",
        "https://api.vinbank.example:/v1/transfers",
        "https://api.vinbank.example/v1/bad path",
    ],
)
def test_egress_rejects_invalid_https_destinations(destination):
    assert is_egress_allowed(destination, "approved transfer amount 500000") is False


@pytest.mark.parametrize(
    "payload",
    [
        "Database host: db.example.net",
        '{"db_host": "postgres"}',
        "Database host db.example.net",
        "Database URL: postgresql://db.example.net/test",
    ],
)
def test_egress_rejects_database_hosts(payload):
    assert is_egress_allowed(APPROVED_URL, payload) is False
