"""
LM Studio client for local models.

Uses OpenAI-compatible API to communicate with LM Studio server.
Supports both orchestrator (nemotron) and worker (phi-4) models.
"""

import time
from typing import Any

from openai import AsyncOpenAI, OpenAI

from src.config import get_settings
from src.models.base import BaseModelClient, ModelConfig, ModelResponse


class LMStudioClient(BaseModelClient):
    """
    Client for LM Studio local models.

    LM Studio provides an OpenAI-compatible API at localhost:1234/v1.
    This client wraps the OpenAI SDK for seamless integration.
    """

    def __init__(
        self,
        model_name: str | None = None,
        config: ModelConfig | None = None,
    ):
        settings = get_settings()

        if config is None:
            config = ModelConfig(
                model_name=model_name or settings.models.orchestrator_model,
            )

        super().__init__(config)

        self._client = OpenAI(
            base_url=settings.lm_studio.base_url,
            api_key=settings.lm_studio.api_key,
        )
        self._async_client = AsyncOpenAI(
            base_url=settings.lm_studio.base_url,
            api_key=settings.lm_studio.api_key,
        )

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def is_local(self) -> bool:
        return True

    @property
    def cost_per_1k_input(self) -> float:
        return 0.0  # Local models are free

    @property
    def cost_per_1k_output(self) -> float:
        return 0.0  # Local models are free

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Send a chat completion request to LM Studio.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            ModelResponse with generated content
        """
        start_time = time.perf_counter()

        request_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stop": kwargs.get("stop", self.config.stop),
        }

        response_format = kwargs.get("response_format")
        if response_format is not None:
            request_params["response_format"] = response_format

        response = self._client.chat.completions.create(**request_params)

        latency_ms = (time.perf_counter() - start_time) * 1000

        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        return ModelResponse(
            content=response.choices[0].message.content or "",
            model=self.model_name,
            usage=usage,
            cost=0.0,  # Local models are free
            latency_ms=latency_ms,
            raw_response=response,
        )

    async def achat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async version of chat."""
        start_time = time.perf_counter()

        request_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stop": kwargs.get("stop", self.config.stop),
        }

        response_format = kwargs.get("response_format")
        if response_format is not None:
            request_params["response_format"] = response_format

        response = await self._async_client.chat.completions.create(**request_params)

        latency_ms = (time.perf_counter() - start_time) * 1000

        usage = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }

        return ModelResponse(
            content=response.choices[0].message.content or "",
            model=self.model_name,
            usage=usage,
            cost=0.0,
            latency_ms=latency_ms,
            raw_response=response,
        )


def get_orchestrator_client() -> LMStudioClient:
    """Get a client configured for the orchestrator model."""
    settings = get_settings()
    return LMStudioClient(model_name=settings.models.orchestrator_model)


def get_phi4_client() -> LMStudioClient:
    """Get a client configured for the Phi-4 model."""
    settings = get_settings()
    return LMStudioClient(model_name=settings.models.phi4_model)
