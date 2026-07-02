"""ベクトルDB初期化のテスト(OPENAI_API_KEYが必要)。最新バージョンのみ。"""

import pytest

from atlas import Atlas


@pytest.mark.integration
class TestVectorDbInitialization:
    """ベクトルDBの初期化が正常に完了し、検索可能な状態になることを確認する。"""

    def test_initialize_creates_collections(self) -> None:
        atlas = Atlas(initialize_vector=True)
        assert atlas.technique_chroma_collection is not None
        assert atlas.casestudy_chroma_collection is not None

    def test_reinitialize_succeeds(self) -> None:
        Atlas(initialize_vector=True)  # 1回目の初期化
        atlas = Atlas(initialize_vector=True)  # 再初期化
        assert atlas.technique_chroma_collection is not None
        assert atlas.casestudy_chroma_collection is not None

    def test_without_initialize_loads_existing(self) -> None:
        Atlas(initialize_vector=True)
        atlas = Atlas(initialize_vector=False)
        assert atlas.technique_chroma_collection is not None
        assert atlas.casestudy_chroma_collection is not None

    def test_search_after_initialize(self) -> None:
        atlas = Atlas(initialize_vector=True)
        results = atlas.search_relevant_technique(query="LLM prompt injection", top_k=3)
        assert len(results) == 3
        cs_results = atlas.search_relevant_casestudy(query="adversarial attack", top_k=3)
        assert len(cs_results) == 3
