from __future__ import annotations

from typing import Generator
import ollama

from src.config import settings


class OllamaClient:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or settings.ollama_model
        self._client = ollama.Client(host=base_url or settings.ollama_base_url)

    def stream_chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        num_predict: int = -1,
        num_ctx: int | None = None,
        seed: int | None = None,
    ) -> Generator[str, None, None]:
        num_ctx = num_ctx or settings.max_context_tokens
        options = {
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "stop": ["<|im_end|>", "</s>", "<|endoftext|>"],
        }
        if seed is not None:
            options["seed"] = seed
        stream = self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            think=False,
            options=options,
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        num_predict: int = 200,
    ) -> str:
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            think=False,
            options={
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": settings.max_context_tokens,
            },
        )
        return resp["message"]["content"]

    def ping(self) -> bool:
        try:
            self._client.list()
            return True
        except Exception:
            return False
