"""ベクトル検索機能のテスト(OPENAI_API_KEYが必要)。"""

import pytest

from atlas.core import Atlas
from atlas.entities import AtlasCaseStudyStep, AtlasTechnique


@pytest.mark.integration
class TestTechniqueVectorSearch:
    """テクニックのベクトル検索テスト。"""

    def test_search_returns_results(self, atlas_with_vector: Atlas) -> None:
        results = atlas_with_vector.search_relevant_technique(query="phishing LLM cloud", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_search_results_are_techniques(self, atlas_with_vector: Atlas) -> None:
        results = atlas_with_vector.search_relevant_technique(query="phishing LLM cloud", top_k=3)
        for tec in results:
            assert isinstance(tec, AtlasTechnique)
            assert tec.id is not None
            assert tec.name is not None
            assert tec.description is not None

    def test_search_with_parent_filter(self, atlas_with_vector: Atlas) -> None:
        results = atlas_with_vector.search_relevant_technique(query="data poisoning", top_k=3, filter="parent")
        assert len(results) == 3
        for tec in results:
            assert not tec.have_parent

    def test_search_with_child_filter(self, atlas_with_vector: Atlas) -> None:
        results = atlas_with_vector.search_relevant_technique(query="data poisoning", top_k=3, filter="child")
        assert len(results) == 3
        for tec in results:
            assert tec.have_parent


@pytest.mark.integration
class TestCaseStudyVectorSearch:
    """ケーススタディのベクトル検索テスト。"""

    def test_search_returns_results(self, atlas_with_vector: Atlas) -> None:
        results = atlas_with_vector.search_relevant_casestudy(query="RAG LLM API", top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_search_results_are_steps(self, atlas_with_vector: Atlas) -> None:
        results = atlas_with_vector.search_relevant_casestudy(query="RAG LLM API", top_k=3)
        for step in results:
            assert isinstance(step, AtlasCaseStudyStep)
            assert step.id is not None
            assert step.technique is not None
            assert step.tactic is not None
