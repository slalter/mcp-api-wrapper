"""Tests for the agent-to-agent message queue."""

from __future__ import annotations

import asyncio

import pytest

from mcp_api_wrapper.queue.message_queue import InMemoryMessageQueue
from mcp_api_wrapper.schemas import (
    APIEvolutionRequest,
    APIEvolutionResponse,
    EvolutionRequestStatus,
)


class TestMessageQueue:
    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self, message_queue: InMemoryMessageQueue) -> None:
        request = APIEvolutionRequest(
            request_id="req-1",
            from_client="client-42",
            request_type="add_endpoint",
            description="Add GET /activity endpoint",
        )
        await message_queue.publish("requests", request, "client_to_backend")

        msg = await message_queue.subscribe("requests", timeout=1.0)
        assert msg is not None
        assert isinstance(msg.payload, APIEvolutionRequest)
        assert msg.payload.request_id == "req-1"
        assert msg.acknowledged

    @pytest.mark.asyncio
    async def test_subscribe_timeout(self, message_queue: InMemoryMessageQueue) -> None:
        msg = await message_queue.subscribe("empty-channel", timeout=0.1)
        assert msg is None

    @pytest.mark.asyncio
    async def test_request_status_tracking(self, message_queue: InMemoryMessageQueue) -> None:
        request = APIEvolutionRequest(
            request_id="req-2",
            from_client="client-1",
            request_type="add_endpoint",
            description="Add DELETE /users/{id}",
        )
        await message_queue.publish("requests", request, "client_to_backend")
        assert message_queue.get_request_status("req-2") == EvolutionRequestStatus.PENDING

        response = APIEvolutionResponse(
            request_id="req-2",
            status=EvolutionRequestStatus.COMPLETED,
            message="Endpoint added and deployed",
            new_version="2.1.0",
            new_endpoints=["DELETE /users/{id}"],
        )
        await message_queue.publish("responses", response, "backend_to_client")
        assert message_queue.get_request_status("req-2") == EvolutionRequestStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_message_history(self, message_queue: InMemoryMessageQueue) -> None:
        for i in range(3):
            req = APIEvolutionRequest(
                request_id=f"req-{i}",
                from_client="client-1",
                request_type="add_endpoint",
                description=f"Request {i}",
            )
            await message_queue.publish("requests", req, "client_to_backend")

        history = message_queue.get_history("requests")
        assert len(history) == 3

        all_history = message_queue.get_history()
        assert len(all_history) == 3

    @pytest.mark.asyncio
    async def test_pending_count(self, message_queue: InMemoryMessageQueue) -> None:
        req = APIEvolutionRequest(
            request_id="req-x",
            from_client="client-1",
            request_type="add_endpoint",
            description="Test",
        )
        await message_queue.publish("ch1", req, "client_to_backend")
        await message_queue.publish("ch1", req, "client_to_backend")
        assert message_queue.pending_count("ch1") == 2

        await message_queue.subscribe("ch1", timeout=0.1)
        assert message_queue.pending_count("ch1") == 1

    @pytest.mark.asyncio
    async def test_rejection_status(self, message_queue: InMemoryMessageQueue) -> None:
        request = APIEvolutionRequest(
            request_id="req-rejected",
            from_client="client-1",
            request_type="other",
            description="Something impossible",
        )
        await message_queue.publish("requests", request, "client_to_backend")

        response = APIEvolutionResponse(
            request_id="req-rejected",
            status=EvolutionRequestStatus.REJECTED,
            message="This change is not feasible",
        )
        await message_queue.publish("responses", response, "backend_to_client")
        assert message_queue.get_request_status("req-rejected") == EvolutionRequestStatus.REJECTED
