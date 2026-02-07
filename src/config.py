"""
Configuration management for Tool Orchestra.

Uses pydantic-settings for type-safe environment variable loading.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LMStudioSettings(BaseSettings):
    """LM Studio local model server settings."""

    model_config = SettingsConfigDict(env_prefix="LM_STUDIO_")

    base_url: str = Field(
        default="http://localhost:1234/v1",
        description="LM Studio API base URL",
    )
    api_key: str = Field(
        default="lm-studio",
        description="API key (usually not required for local)",
    )


class ModelSettings(BaseSettings):
    """Model configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    orchestrator_model: str = Field(
        default="nemotron-orchestrator-8b",
        alias="ORCHESTRATOR_MODEL",
        description="Orchestrator model name in LM Studio",
    )
    phi4_model: str = Field(
        default="microsoft_phi-4-mini-instruct",
        alias="PHI4_MODEL",
        description="Phi-4 model name in LM Studio",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash-exp",
        alias="GEMINI_MODEL",
        description="Gemini model name",
    )
    gemini_api_key: str = Field(
        default="",
        alias="GEMINI_API_KEY",
        description="Google Gemini API key",
    )


class PreferenceDefaults(BaseSettings):
    """Default preference vector values."""

    model_config = SettingsConfigDict(env_prefix="DEFAULT_")

    budget: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default budget preference (0=cheap, 1=quality)",
    )
    privacy: bool = Field(
        default=False,
        description="Default privacy mode (True=local only)",
    )
    quality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default quality preference",
    )


class BraveSearchSettings(BaseSettings):
    """Brave Search API configuration."""

    model_config = SettingsConfigDict(env_prefix="BRAVE_", env_file=".env", extra="ignore")

    api_key: str = Field(
        default="",
        description="Brave Search API key",
    )
    base_url: str = Field(
        default="https://api.search.brave.com/res/v1/web/search",
        description="Brave Search API endpoint",
    )


class VectorStoreSettings(BaseSettings):
    """Vector store configuration."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_")

    type: Literal["faiss", "chromadb"] = Field(
        default="faiss",
        description="Vector store type",
    )
    path: Path = Field(
        default=Path("./data/vectorstore"),
        description="Vector store data path",
    )
    embedding_model: str = Field(
        default="gemini-2.0-flash",
        description="Model to use for generating embeddings",
    )
    chunk_size: int = Field(
        default=500,
        description="Document chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between chunks in characters",
    )


class CacheSettings(BaseSettings):
    """Response caching configuration."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    enabled: bool = Field(
        default=True,
        description="Enable response caching",
    )
    dir: Path = Field(
        default=Path(".cache/responses"),
        description="Cache directory",
    )
    ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds",
    )


class LangSmithSettings(BaseSettings):
    """LangSmith tracing configuration."""

    model_config = SettingsConfigDict(env_prefix="LANGCHAIN_")

    tracing_v2: bool = Field(
        default=False,
        alias="LANGCHAIN_TRACING_V2",
        description="Enable LangSmith tracing",
    )
    api_key: str = Field(
        default="",
        alias="LANGCHAIN_API_KEY",
        description="LangSmith API key",
    )
    project: str = Field(
        default="tool-orchestra",
        alias="LANGCHAIN_PROJECT",
        description="LangSmith project name",
    )


class APISettings(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    port: int = Field(
        default=8000,
        description="API server port",
    )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Safety limits
    max_iterations: int = Field(
        default=10,
        description="Maximum iterations per query",
    )

    # Sub-settings
    lm_studio: LMStudioSettings = Field(default_factory=LMStudioSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    preferences: PreferenceDefaults = Field(default_factory=PreferenceDefaults)
    brave_search: BraveSearchSettings = Field(default_factory=BraveSearchSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    api: APISettings = Field(default_factory=APISettings)

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"

    @property
    def knowledge_dir(self) -> Path:
        """Get knowledge base directory."""
        return self.data_dir / "knowledge"

    @property
    def synthetic_dir(self) -> Path:
        """Get synthetic data directory."""
        return self.data_dir / "synthetic"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience function for quick access
settings = get_settings()


def setup_langsmith() -> None:
    """Configure LangSmith tracing if enabled."""
    import os

    cfg = get_settings()
    if cfg.langsmith.tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith.api_key
        os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith.project
        print(f"LangSmith tracing enabled for project: {cfg.langsmith.project}")


if __name__ == "__main__":
    # Print current configuration (useful for debugging)

    cfg = get_settings()
    print("Current Configuration:")
    print("-" * 40)
    print(f"Log Level: {cfg.log_level}")
    print(f"Max Iterations: {cfg.max_iterations}")
    print(f"LM Studio URL: {cfg.lm_studio.base_url}")
    print(f"Orchestrator Model: {cfg.models.orchestrator_model}")
    print(f"Phi-4 Model: {cfg.models.phi4_model}")
    print(f"Gemini Model: {cfg.models.gemini_model}")
    print(f"Cache Enabled: {cfg.cache.enabled}")
    print(f"LangSmith Tracing: {cfg.langsmith.tracing_v2}")
