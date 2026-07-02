import pytest
from dotenv import dotenv_values

from atlas import Atlas

_env = dotenv_values(".env.dev")


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """integrationテスト実行前にOPENAI_API_KEYを設定する。"""
    api_key = _env.get("OPENAI_API_KEY") or ""
    monkeypatch.setenv("OPENAI_API_KEY", api_key)


@pytest.fixture
def atlas_with_vector() -> Atlas:
    """ベクトルDB付きのAtlasインスタンス(最新版)。integrationテスト用。"""
    return Atlas(initialize_vector=True)
