from __future__ import annotations

from src.config import settings
from src.llm.client import OllamaClient


class FakeOllamaClient:
    def __init__(self) -> None:
        self.request: dict | None = None

    def chat(self, **kwargs: object) -> object:
        self.request = kwargs
        if kwargs.get("stream"):
            return iter([{"message": {"content": "ok"}}])
        return {"message": {"content": "ok"}}


def test_chat_uses_configured_context_size() -> None:
    transport = FakeOllamaClient()
    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client._client = transport

    result = client.chat([{"role": "user", "content": "test"}])

    assert result == "ok"
    assert transport.request is not None
    options = transport.request["options"]
    assert isinstance(options, dict)
    assert options["num_ctx"] == settings.max_context_tokens


def test_stream_chat_passes_optional_seed() -> None:
    transport = FakeOllamaClient()
    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client._client = transport

    result = "".join(
        client.stream_chat(
            [{"role": "user", "content": "test"}],
            seed=42,
        )
    )

    assert result == "ok"
    assert transport.request is not None
    options = transport.request["options"]
    assert isinstance(options, dict)
    assert options["seed"] == 42


def test_stream_chat_omits_seed_by_default() -> None:
    transport = FakeOllamaClient()
    client = OllamaClient.__new__(OllamaClient)
    client.model = "test-model"
    client._client = transport

    assert "".join(client.stream_chat([{"role": "user", "content": "test"}])) == "ok"

    assert transport.request is not None
    options = transport.request["options"]
    assert isinstance(options, dict)
    assert "seed" not in options
