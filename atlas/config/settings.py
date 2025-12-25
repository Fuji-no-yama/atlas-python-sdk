from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """
    全体設定
    """

    model_config = SettingsConfigDict(env_file=".env.dev", env_file_encoding="utf-8")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    available_versions: list[str] = Field(
        default_factory=lambda: ["4.4.0", "4.5.0", "4.6.0", "4.7.0", "4.8.0", "4.9.0", "5.0.0", "5.1.0"],
        alias="ATLAS_AVAILABLE_VERSIONS",
    )
    latest_version: str = Field("5.1.0", alias="ATLAS_LATEST_VERSION")

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
