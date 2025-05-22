from functools import lru_cache

from pydantic_settings import BaseSettings


@lru_cache
class Settings(BaseSettings):
    PATRA_URL: str | None = None
    HTTPX_TRUST_ENV: bool = True


    class Config:
        env_file = ".env"

config = Settings()