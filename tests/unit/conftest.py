import os

import pytest

from atlas import Atlas

# 非常に古いリリース(mitigations=0など)ではnot-emptyアサーションが落ちるため、
# フル装備が揃っている 2025.09 以降のみを全リリース検証の対象とする。
_SUPPORTED_RELEASES: list[str] = [
    "2025.09",
    "2025.10",
    "2025.11",
    "2025.11.2",
    "2025.12",
    "2026.01",
    "2026.02",
    "2026.03",
    "2026.04",
    "2026.05",
    "2026.06",
]


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
    """全リリース検証用にサポート対象のリリース識別子リストを返す。"""
    os.environ.pop("OPENAI_API_KEY", None)
    return list(_SUPPORTED_RELEASES)
