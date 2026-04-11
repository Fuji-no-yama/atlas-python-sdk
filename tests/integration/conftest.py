import pytest

from atlas.core import Atlas


@pytest.fixture(scope="module")
def atlas_with_vector() -> Atlas:
    """ベクトルDB付きのAtlasインスタンス(最新版)。integrationテスト用。"""
    return Atlas(initialize_vector=True)
