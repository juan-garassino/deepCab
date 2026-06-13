"""Single source of truth for runtime settings. Replaces every os.environ.get site
in the legacy package atomically. Per-env loading via DEEPCAB_ENV (or its legacy
alias APP_ENV) = {local,dev,staging,prod}."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import AliasChoices, Field
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    SettingsConfigDict,
)

from deepCab.schemas.enums import AppEnv, DataSource, ModelTarget


def _env_file() -> str:
    # DEEPCAB_ENV is the documented public name; APP_ENV remains as a legacy
    # alias for any caller still setting it. First non-empty wins.
    env = os.environ.get("DEEPCAB_ENV") or os.environ.get("APP_ENV") or "dev"
    return f".env.{env}"


class FileUriEnvSettingsSource(EnvSettingsSource):
    """Source-level resolver for the docker-secrets `file:` URI convention.

    Compose pattern:
        secrets:
          deepcab_api_key:
            file: ./secrets/deepcab_api_key
        services:
          api:
            secrets: [deepcab_api_key]
            environment:
              DEEPCAB_API_KEY: file:/run/secrets/deepcab_api_key

    When the resolved env value starts with `file:`, this source reads the
    referenced file and substitutes its (stripped) contents. Plain values pass
    through untouched. Resolution happens once at the source layer instead of
    per-field via `@field_validator`, so every string field benefits without
    boilerplate.
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        value, key, is_complex = super().get_field_value(field, field_name)
        if isinstance(value, str) and value.startswith("file:"):
            path = Path(value[len("file:") :]).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"secret file not found: {path}")
            value = path.read_text().strip()
        return value, key, is_complex


def _customise_sources(
    settings_cls,
    init_settings,
    env_settings,
    dotenv_settings,
    file_secret_settings,
):
    """Replace the default env source with `FileUriEnvSettingsSource` so any
    `file:` URI in env (or `.env`) is resolved before assignment."""
    return (
        init_settings,
        FileUriEnvSettingsSource(settings_cls),
        dotenv_settings,
        file_secret_settings,
    )


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATA_", env_file=_env_file(), extra="ignore")

    source: DataSource = DataSource.LOCAL
    local_path: Path = Path.home() / ".lewagon" / "mlops" / "data"
    parquet_path: Path = Path.home() / ".lewagon" / "mlops" / "data" / "parquet"
    chunk_size: int = 2000
    dataset_size: str = "1k"
    validation_dataset_size: str = "1k"

    # BigQuery ingest target (only consumed when source == DataSource.QUERY).
    # Defaults match the garassino-ml dev project; override per env.
    bq_project: str = "garassino-ml"
    bq_dataset: str = "taxi"
    bq_table: str = "yellow_trips_raw"
    # Project the BQ job bills to. Differs from bq_project when reading a public
    # dataset (e.g. bq_project=bigquery-public-data, billed to our own project).
    # None → bills to bq_project (same-project read). Env: DATA_BQ_BILLING_PROJECT.
    bq_billing_project: str | None = None
    # Optional WHERE clause applied to the BQ read; set per chunk by the
    # simulate flow via the DATA_BQ_WHERE env var. None means "no filter"
    # (preprocess.load falls back to LIMIT-only).
    bq_where: str | None = None


class RegistrySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REGISTRY_", env_file=_env_file(), extra="ignore")

    target: ModelTarget = ModelTarget.LOCAL
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
    # Optional Telegram bot credentials. Same `file:` URI convention as Slack.
    # Both must be set for `obs.telegram.post()` to fire; either missing → no-op.
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        return _customise_sources(
            settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
        )


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=_env_file(), extra="ignore")

    api_key: str | None = None
    model: str = "gpt-4o-mini"

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        return _customise_sources(
            settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
        )


class DeepCabSettings(BaseSettings):
    """deepCab-specific settings — including the API key that gates training
    + agent endpoints. Separate from OPENAI_API_KEY (Phase 8 conflated them;
    P13 untangles)."""

    model_config = SettingsConfigDict(env_prefix="DEEPCAB_", env_file=_env_file(), extra="ignore")

    api_key: str | None = None

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        return _customise_sources(
            settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
        )


class Settings(BaseSettings):
    """Composite root. Sub-settings hold their own env prefixes."""

    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    app_env: Annotated[
        AppEnv,
        Field(
            default=AppEnv.DEV,
            validation_alias=AliasChoices("DEEPCAB_ENV", "APP_ENV"),
        ),
    ]
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
