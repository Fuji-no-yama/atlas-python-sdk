import os

import pytest

from atlas.core import Atlas


@pytest.fixture(autouse=True)
def _unset_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """unitテスト実行時はOPENAI_API_KEYを除去してベクトルDB初期化をスキップする。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(scope="module")
def atlas_default() -> Atlas:
    """デフォルトバージョンのAtlasインスタンス(ベクトルDBなし)。"""
    os.environ.pop("OPENAI_API_KEY", None)
    return Atlas()


@pytest.fixture(scope="module")
def all_versions() -> list[str]:
    """利用可能な全バージョンのリスト。"""
    os.environ.pop("OPENAI_API_KEY", None)
    return Atlas().get_available_versions()
