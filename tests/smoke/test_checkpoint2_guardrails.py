"""Checkpoint 2 regression tests with no provider calls."""

import asyncio

import pytest
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from core.utils import chat_with_agent
from guardrails.input_guardrails import InputGuardrailPlugin, topic_filter
from guardrails.output_guardrails import OutputGuardrailPlugin, content_filter


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Tóm tắt email chuyển khoản delay", "ALLOW"),
        ("Which accounts support recurring payments?", "ALLOW"),
        ("I am interested in cooking pasta.", "BLOCK"),
    ],
)
def test_topic_filter_matches_banking_terms(prompt, expected):
    assert topic_filter(prompt) == expected


def test_content_filter_redacts_password_without_delimiter():
    filtered = content_filter("Temporary password example-test-456.")

    assert not filtered["safe"]
    assert filtered["issues"] == ["password: 1 found"]
    assert filtered["redacted"] == "Temporary [REDACTED]."


def test_adk_stops_blocked_inputs_before_model_and_keeps_them_out_of_history():
    model_requests = []

    class StubModel(BaseLlm):
        model: str = "checkpoint2-stub"

        async def generate_content_async(self, llm_request, stream=False):
            model_requests.append(llm_request)
            yield LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Temporary password example-test-456.")],
                )
            )

    async def run():
        guard = InputGuardrailPlugin()
        agent = Agent(name="checkpoint2_test", model=StubModel())
        runner = InMemoryRunner(
            agent=agent,
            app_name="checkpoint2_test",
            plugins=[guard, OutputGuardrailPlugin(use_llm_judge=False)],
        )
        blocked_prompt = "I am interested in cooking pasta."
        try:
            refusal, session = await chat_with_agent(agent, runner, blocked_prompt)
            assert "banking services" in refusal
            assert not model_requests

            response, _ = await chat_with_agent(
                agent, runner, "Tóm tắt email chuyển khoản delay", session_id=session.id
            )
            assert response == "Temporary [REDACTED]."
            assert (guard.blocked_count, guard.total_count) == (1, 2)
            assert len(model_requests) == 1
            assert all(
                blocked_prompt not in (part.text or "")
                for content in model_requests[0].contents
                for part in content.parts or []
            )
        finally:
            await runner.close()

    asyncio.run(run())
