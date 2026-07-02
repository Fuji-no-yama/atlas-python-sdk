from dotenv import load_dotenv
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """
    全体設定。バージョン一覧は manifest.yaml から動的に解決するためここには持たない。
    """

    model_config = SettingsConfigDict(env_file=".env.dev", env_file_encoding="utf-8")

    # .env > init kwargs > OS env の優先順位を維持
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type["Settings"],  # noqa: ARG003
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return dotenv_settings, init_settings, env_settings, file_secret_settings


load_dotenv(override=True)
settings = Settings()
