import asyncio
import os
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel, Field, SecretStr, model_validator
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMConfig(BaseModel):
    provider: Provider
    model: str
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    max_concurrency: int = Field(default=10, gt=0)
    timeout: float = Field(default=30.0, gt=0)  # segundos
    max_tokens: int = Field(default=1024, gt=0)

    @model_validator(mode="after")
    def validate_api_key(self) -> "LLMConfig":
        key = self._key_for(self.provider)
        if key is None or not key.get_secret_value().strip():
            raise ValueError(f"Falta la API key para el proveedor '{self.provider.value}'")
        return self

    def _key_for(self, provider: Provider) -> SecretStr | None:
        return {
            Provider.OPENAI: self.openai_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
        }[provider]

    @property
    def api_key(self) -> str:
        return self._key_for(self.provider).get_secret_value()


class BaseLLMClient(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def chat(self, prompt: str) -> str:
        async with self._semaphore:
            return await self._chat(prompt)

    @abstractmethod
    async def _chat(self, prompt: str) -> str: ...

    @abstractmethod
    async def close(self) -> None: ...


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config.api_key, timeout=config.timeout)

    async def _chat(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    async def close(self) -> None:
        await self.client.close()


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncAnthropic(api_key=config.api_key, timeout=config.timeout)

    async def _chat(self, prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.config.max_tokens,  # obligatorio en Anthropic
            messages=[{"role": "user", "content": prompt}],
        )

        return "".join(block.text for block in response.content if block.type == "text")

    async def close(self) -> None:
        await self.client.close()


class LLMFactory:
    _registry: dict[Provider, type[BaseLLMClient]] = {
        Provider.OPENAI: OpenAIClient,
        Provider.ANTHROPIC: AnthropicClient,
    }

    @classmethod
    def create_client(cls, config: LLMConfig) -> BaseLLMClient:
        try:
            return cls._registry[config.provider](config)
        except KeyError:
            raise ValueError(f"Proveedor no soportado: {config.provider}")


async def main():
    config = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_concurrency=10,
        timeout=10,
    )
    client = LLMFactory.create_client(config)
    try:
        respuesta = await client.chat("Hello, how are you?")
        print(respuesta)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())