"""Message queue for agent-to-agent communication.

Enables client agents to request API changes and receive responses
from the backend agent that manages the API repository.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from mcp_api_wrapper.schemas import APIEvolutionRequest, APIEvolutionResponse, EvolutionRequestStatus


class QueueMessage(BaseModel):
    """Envelope for messages in the queue."""

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    channel: str = Field(description="Routing channel")
    direction: Literal["client_to_backend", "backend_to_client"] = Field(
        description="Message direction"
    )
    payload: APIEvolutionRequest | APIEvolutionResponse
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False


class InMemoryMessageQueue:
    """In-memory message queue for development/testing.

    For production, swap with Redis Streams, RabbitMQ, or similar.
    The interface stays the same.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[QueueMessage]] = defaultdict(asyncio.Queue)
        self._history: list[QueueMessage] = []
        self._request_status: dict[str, EvolutionRequestStatus] = {}

    async def publish(
        self,
        channel: str,
        payload: APIEvolutionRequest | APIEvolutionResponse,
        direction: Literal["client_to_backend", "backend_to_client"],
    ) -> QueueMessage:
        """Publish a message to a channel."""
        msg = QueueMessage(
            channel=channel,
            direction=direction,
            payload=payload,
        )
        await self._queues[channel].put(msg)
        self._history.append(msg)

        # Track request status
        if isinstance(payload, APIEvolutionRequest):
            self._request_status[payload.request_id] = EvolutionRequestStatus.PENDING
        elif isinstance(payload, APIEvolutionResponse):
            self._request_status[payload.request_id] = payload.status

        return msg

    async def subscribe(
        self,
        channel: str,
        timeout: float = 30.0,
    ) -> QueueMessage | None:
        """Wait for a message on a channel. Returns None on timeout."""
        try:
            msg = await asyncio.wait_for(
                self._queues[channel].get(),
                timeout=timeout,
            )
            msg.acknowledged = True
            return msg
        except asyncio.TimeoutError:
            return None

    def get_request_status(self, request_id: str) -> EvolutionRequestStatus | None:
        """Check the status of a previously submitted request."""
        return self._request_status.get(request_id)

    def get_history(
        self,
        channel: str | None = None,
        limit: int = 50,
    ) -> list[QueueMessage]:
        """Get message history, optionally filtered by channel."""
        msgs = self._history
        if channel:
            msgs = [m for m in msgs if m.channel == channel]
        return msgs[-limit:]

    def pending_count(self, channel: str) -> int:
        """Number of unprocessed messages in a channel."""
        return self._queues[channel].qsize()
