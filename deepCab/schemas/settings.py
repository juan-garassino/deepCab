"""Single source of truth for runtime settings. Replaces every os.environ.get site
in the legacy package atomically. Per-env loading via APP_ENV={dev,staging,prod}."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_file() -> str:
    env = os.environ.get("APP_ENV", "dev")
    return f".env.{env}"


def _maybe_read_file(value: str | None) -> str | None:
    """If the env value starts with `file:` (docker secrets convention) or
    points at a readable file, return the file content stripped. Otherwise
    pass the raw value through unchanged.

    Compose pattern:
        secrets:
          deepcab_api_key:
            file: ./secrets/deepcab_api_key
        services:
          api:
            secrets: [deepcab_api_key]
            environment:
              DEEPCAB_API_KEY: file:/run/secrets/deepcab_api_key

    On host startup the file appears at `/run/secrets/deepcab_api_key`;
    pydantic-settings reads the env var, this helper detects the `file:` prefix
    and substitutes the file content. Never logs the value.
    """
    if value is None:
        return None
    if value.startswith("file:"):
        path = Path(value[len("file:"):])
        if not path.exists():
            raise FileNotFoundError(f"secret file not found: {path}")
        return path.read_text().strip()
    return value


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_", env_file=_env_file(), extra="ignore")

    source: Literal["local", "query", "cloud"] = "local"
    local_path: Path = Path.home() / ".lewagon" / "mlops" / "data"
    parquet_path: Path = Path.home() / ".lewagon" / "mlops" / "data" / "parquet"
    chunk_size: int = 2000
    dataset_size: str = "1k"
    validation_dataset_size: str = "1k"


class RegistrySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REGISTRY_", env_file=_env_file(), extra="ignore")

    target: Literal["local", "gcs", "mlflow"] = "local"
    local_path: Path = Path.home() / ".lewagon" / "mlops" / "training_outputs"
    bucket_name: str | None = None


class GCPSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GCP_", env_file=_env_file(), extra="ignore")

    project_id: str | None = None
    dataset: str | None = None


class MLflowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLFLOW_", env_file=_env_file(), extra="ignore")

    tracking_uri: str | None = None
    experiment: str | None = None
    model_name: str | None = None
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"


class ObsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OBS_", env_file=_env_file(), extra="ignore")

    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "deepcab"
    prom_port: int = 9090
    trace_dir: Path = Path("traces")
    trace_enabled: bool = True
    redis_url: str | None = None  # opt-in Redis for SHAP cache + future state share
    # Comma-separated origins, e.g. "http://localhost:3000,https://app.example.com".
    # Empty in prod forces explicit allowlist; "*" allowed in dev only.
    cors_allow_origins: str = "http://localhost:3000,http://localhost:8501"
    # Optional Slack incoming-webhook URL. Resolved via the `file:` docker-secrets
    # pattern (set OBS_SLACK_WEBHOOK_URL=file:/run/secrets/slack_webhook_url).
    # Empty/None means the in-process slack helper is a no-op.
    slack_webhook_url: str | None = None

    @field_validator("slack_webhook_url", mode="before")
    @classmethod
    def _read_file(cls, v):
        return _maybe_read_file(v)


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=_env_file(), extra="ignore")

    api_key: str | None = None
    model: str = "gpt-4o-mini"

    @field_validator("api_key", mode="before")
    @classmethod
    def _read_file(cls, v):
        return _maybe_read_file(v)


class DeepCabSettings(BaseSettings):
    """deepCab-specific settings — including the API key that gates training
    + agent endpoints. Separate from OPENAI_API_KEY (Phase 8 conflated them;
    P13 untangles)."""

    model_config = SettingsConfigDict(env_prefix="DEEPCAB_", env_file=_env_file(), extra="ignore")

    api_key: str | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def _read_file(cls, v):
        return _maybe_read_file(v)


class Settings(BaseSettings):
    """Composite root. Sub-settings hold their own env prefixes."""

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    app_env: Literal["dev", "staging", "prod"] = Field(
        default="dev", validation_alias="APP_ENV"
    )
    data: DataSettings = Field(default_factory=DataSettings)
    registry: RegistrySettings = Field(default_factory=RegistrySettings)
    gcp: GCPSettings = Field(default_factory=GCPSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    obs: ObsSettings = Field(default_factory=ObsSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    deepcab: DeepCabSettings = Field(default_factory=DeepCabSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
