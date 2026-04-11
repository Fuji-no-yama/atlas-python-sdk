"""ID検索機能のテスト(ベクトル検索を除く)。"""

import pytest

from atlas.core import Atlas
from atlas.entities import AtlasCaseStudy, AtlasCaseStudyStep, AtlasMitigation, AtlasTactic, AtlasTechnique


class TestSearchTechniqueById:
    """テクニックID検索のテスト。"""

    def test_search_existing_technique(self, atlas_default: Atlas) -> None:
        first = atlas_default.technique_list[0]
        result = atlas_default.search_tec_from_id(tec_id=first.id)
        assert isinstance(result, AtlasTechnique)
        assert result.id == first.id
        assert result.name == first.name

    def test_search_nonexistent_technique_raises(self, atlas_default: Atlas) -> None:
        with pytest.raises(ValueError, match="見つかりませんでした"):
            atlas_default.search_tec_from_id(tec_id="AML.T9999")


class TestSearchTacticById:
    """タクティックID検索のテスト。"""

    def test_search_existing_tactic(self, atlas_default: Atlas) -> None:
        first = atlas_default.tactic_list[0]
        result = atlas_default.search_tac_from_id(tac_id=first.id)
        assert isinstance(result, AtlasTactic)
        assert result.id == first.id

    def test_search_nonexistent_tactic_raises(self, atlas_default: Atlas) -> None:
        with pytest.raises(ValueError, match="見つかりませんでした"):
            atlas_default.search_tac_from_id(tac_id="AML.TA9999")


class TestSearchMitigationById:
    """緩和策ID検索のテスト。"""

    def test_search_existing_mitigation(self, atlas_default: Atlas) -> None:
        first = atlas_default.mitigation_list[0]
        result = atlas_default.search_mit_from_id(mit_id=first.id)
        assert isinstance(result, AtlasMitigation)
        assert result.id == first.id

    def test_search_nonexistent_mitigation_raises(self, atlas_default: Atlas) -> None:
        with pytest.raises(ValueError, match="見つかりませんでした"):
            atlas_default.search_mit_from_id(mit_id="AML.M9999")


class TestSearchCaseStudyById:
    """ケーススタディID検索のテスト。"""

    def test_search_existing_casestudy(self, atlas_default: Atlas) -> None:
        first = atlas_default.casestudy_list[0]
        result = atlas_default.search_cs_from_id(cs_id=first.id)
        assert isinstance(result, AtlasCaseStudy)
        assert result.id == first.id

    def test_search_nonexistent_casestudy_raises(self, atlas_default: Atlas) -> None:
        with pytest.raises(ValueError, match="見つかりませんでした"):
            atlas_default.search_cs_from_id(cs_id="AML.CS9999")


class TestSearchCaseStudyStepById:
    """ケーススタディステップID検索のテスト。"""

    def test_search_existing_step(self, atlas_default: Atlas) -> None:
        first_cs = atlas_default.casestudy_list[0]
        first_step = first_cs.procedure[0]
        result = atlas_default.search_cs_step_from_id(cs_step_id=first_step.id)
        assert isinstance(result, AtlasCaseStudyStep)
        assert result.id == first_step.id

    def test_search_nonexistent_step_raises(self, atlas_default: Atlas) -> None:
        with pytest.raises(ValueError, match="見つかりませんでした"):
            atlas_default.search_cs_step_from_id(cs_step_id="AML.CS9999.99")
