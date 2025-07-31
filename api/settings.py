from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APPLICATIONINSIGHTS_CONNECTION_STRING: str = ""
    RUN_ID: str = ""
    SERVICE_NAME: str = ""
    MLFLOW_TRACKING_URI: str = ""

    model_config = SettingsConfigDict(env_file=".env")

@lru_cache
def get_settings() -> Settings:
    return Settings()