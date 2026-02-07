"""
Gemini API client for cloud-based model access.

This is the "expert" tier - used for complex reasoning tasks
when local models are insufficient.

Uses the google-genai SDK (unified Google GenAI SDK).
"""

import time
from typing import Any

from google import genai
from google.genai import types

from src.config import get_settings
from src.models.base import BaseModelClient, ModelConfig, ModelResponse

# Gemini pricing (as of 2024, subject to change)
# https://ai.google.dev/pricing
GEMINI_PRICING = {
    "gemini-2.5-flash": {
        "input": 0.0,  # Free tier
        "output": 0.0,
    },
    "gemini-1.5-flash": {
        "input": 0.075 / 1000,  # $0.075 per 1M tokens
        "output": 0.30 / 1000,
    },
    "gemini-1.5-pro": {
        "input": 1.25 / 1000,  # $1.25 per 1M tokens
        "output": 5.00 / 1000,
    },
    "gemini-2.0-flash": {
        "input": 0.10 / 1000,
        "output": 0.40 / 1000,
    },
}


class GeminiClient(BaseModelClient):
    """
    Client for Google Gemini API.

    Used as the "expert" model for complex reasoning tasks.
    Incurs API costs, so usage should be optimized.
    """

    def __init__(
        self,
        model_name: str | None = None,
        config: ModelConfig | None = None,
    ):
        settings = get_settings()

        if config is None:
            config = ModelConfig(
                model_name=model_name or settings.models.gemini_model,
            )

        super().__init__(config)

        # Configure the Gemini API client
        api_key = settings.models.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Please set it in your .env file.")

        self._client = genai.Client(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def is_local(self) -> bool:
        return False

    @property
    def cost_per_1k_input(self) -> float:
        pricing = GEMINI_PRICING.get(
            self.model_name,
            {"input": 0.001, "output": 0.002},  # Default conservative estimate
        )
        return pricing["input"] * 1000  # Convert to per 1k tokens

    @property
    def cost_per_1k_output(self) -> float:
        pricing = GEMINI_PRICING.get(
            self.model_name,
            {"input": 0.001, "output": 0.002},
        )
        return pricing["output"] * 1000

    def _build_contents(self, messages: list[dict[str, str]]) -> list[types.Content]:
        """Convert OpenAI-style messages to google-genai Content objects."""
        contents: list[types.Content] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Gemini uses "user" and "model" roles
            if role == "system":
                # Inject system message as a user/model exchange
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=f"System: {content}")])
                )
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="Understood. I will follow these instructions.")],
                    )
                )
            elif role == "assistant":
                contents.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=content)])
                )
            else:  # user
                contents.append(
                    types.Content(role="user", parts=[types.Part.from_text(text=content)])
                )

        return contents

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Send a chat completion request to Gemini.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters

        Returns:
            ModelResponse with generated content
        """
        start_time = time.perf_counter()

        # Build generation config
        gen_config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
            stop_sequences=kwargs.get("stop", self.config.stop),
        )

        # Convert messages to Content objects
        contents = self._build_contents(messages)

        # Generate response
        response = self._client.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            config=gen_config,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract token counts
        input_tokens = 0
        output_tokens = 0

        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        cost = self.calculate_cost(input_tokens, output_tokens)

        return ModelResponse(
            content=response.text or "",
            model=self.model_name,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            raw_response=response,
        )

    async def achat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Async version of chat.

        Uses the google-genai async client.
        """
        start_time = time.perf_counter()

        gen_config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.config.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            top_p=kwargs.get("top_p", self.config.top_p),
            stop_sequences=kwargs.get("stop", self.config.stop),
        )

        contents = self._build_contents(messages)

        response = await self._client.aio.models.generate_content(
            model=self.config.model_name,
            contents=contents,
            config=gen_config,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        input_tokens = 0
        output_tokens = 0

        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        cost = self.calculate_cost(input_tokens, output_tokens)

        return ModelResponse(
            content=response.text or "",
            model=self.model_name,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            raw_response=response,
        )


def get_gemini_client() -> GeminiClient:
    """Get a client configured for the default Gemini model."""
    return GeminiClient()
