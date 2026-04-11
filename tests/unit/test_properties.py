"""Atlasインスタンスの基本プロパティに関するテスト。"""

import pytest

from atlas.core import Atlas
from atlas.entities import AtlasCaseStudy, AtlasMitigation, AtlasTactic, AtlasTechnique


class TestAtlasProperties:
    """各バージョンでAtlasが正しく初期化され、リストが空でないことを確認する。"""

    def test_version_is_set(self, atlas_default: Atlas) -> None:
        assert atlas_default.version is not None
        assert atlas_default.version != ""

    def test_technique_list_not_empty(self, atlas_default: Atlas) -> None:
        assert isinstance(atlas_default.technique_list, list)
        assert len(atlas_default.technique_list) > 0

    def test_mitigation_list_not_empty(self, atlas_default: Atlas) -> None:
        assert isinstance(atlas_default.mitigation_list, list)
        assert len(atlas_default.mitigation_list) > 0

    def test_casestudy_list_not_empty(self, atlas_default: Atlas) -> None:
        assert isinstance(atlas_default.casestudy_list, list)
        assert len(atlas_default.casestudy_list) > 0

    def test_tactic_list_not_empty(self, atlas_default: Atlas) -> None:
        assert isinstance(atlas_default.tactic_list, list)
        assert len(atlas_default.tactic_list) > 0


class TestTechniqueProperties:
    """テクニックオブジェクトのプロパティを確認する。"""

    def test_technique_has_required_fields(self, atlas_default: Atlas) -> None:
        for tec in atlas_default.technique_list:
            assert isinstance(tec, AtlasTechnique)
            assert tec.id is not None and tec.id != ""
            assert tec.name is not None and tec.name != ""
            assert tec.description is not None and tec.description != ""

    def test_technique_id_format(self, atlas_default: Atlas) -> None:
        for tec in atlas_default.technique_list:
            assert tec.id.startswith("AML.T"), f"Unexpected technique id format: {tec.id}"

    def test_child_technique_has_parent(self, atlas_default: Atlas) -> None:
        for tec in atlas_default.technique_list:
            if tec.have_parent:
                assert tec.parent_id is not None, f"Child technique {tec.id} has no parent_id"

    def test_technique_has_tactics(self, atlas_default: Atlas) -> None:
        for tec in atlas_default.technique_list:
            assert isinstance(tec.tactics, list)
            assert len(tec.tactics) > 0, f"Technique {tec.id} has no tactics"


class TestMitigationProperties:
    """緩和策オブジェクトのプロパティを確認する。"""

    def test_mitigation_has_required_fields(self, atlas_default: Atlas) -> None:
        for mit in atlas_default.mitigation_list:
            assert isinstance(mit, AtlasMitigation)
            assert mit.id is not None and mit.id != ""
            assert mit.name is not None and mit.name != ""
            assert mit.description is not None and mit.description != ""

    def test_mitigation_id_format(self, atlas_default: Atlas) -> None:
        for mit in atlas_default.mitigation_list:
            assert mit.id.startswith("AML.M"), f"Unexpected mitigation id format: {mit.id}"

    def test_mitigation_has_techniques(self, atlas_default: Atlas) -> None:
        for mit in atlas_default.mitigation_list:
            assert isinstance(mit.technique_list, list)
            assert len(mit.technique_list) > 0, f"Mitigation {mit.id} has no techniques"


class TestCaseStudyProperties:
    """ケーススタディオブジェクトのプロパティを確認する。"""

    def test_casestudy_has_required_fields(self, atlas_default: Atlas) -> None:
        for cs in atlas_default.casestudy_list:
            assert isinstance(cs, AtlasCaseStudy)
            assert cs.id is not None and cs.id != ""
            assert cs.name is not None and cs.name != ""
            assert cs.summary is not None and cs.summary != ""

    def test_casestudy_id_format(self, atlas_default: Atlas) -> None:
        for cs in atlas_default.casestudy_list:
            assert cs.id.startswith("AML.CS"), f"Unexpected casestudy id format: {cs.id}"

    def test_casestudy_has_procedure(self, atlas_default: Atlas) -> None:
        for cs in atlas_default.casestudy_list:
            assert isinstance(cs.procedure, list)
            assert len(cs.procedure) > 0, f"CaseStudy {cs.id} has no procedure steps"

    def test_casestudy_type(self, atlas_default: Atlas) -> None:
        for cs in atlas_default.casestudy_list:
            assert cs.type in ("exercise", "incident"), f"Unexpected casestudy type: {cs.type}"


class TestTacticProperties:
    """タクティックオブジェクトのプロパティを確認する。"""

    def test_tactic_has_required_fields(self, atlas_default: Atlas) -> None:
        for tac in atlas_default.tactic_list:
            assert isinstance(tac, AtlasTactic)
            assert tac.id is not None and tac.id != ""
            assert tac.name is not None and tac.name != ""

    def test_tactic_id_format(self, atlas_default: Atlas) -> None:
        for tac in atlas_default.tactic_list:
            assert tac.id.startswith("AML.TA"), f"Unexpected tactic id format: {tac.id}"


class TestAllVersions:
    """全バージョンでAtlasが正常に初期化されることを確認する。"""

    def test_all_versions_loadable(self, all_versions: list[str]) -> None:
        for version in all_versions:
            atlas = Atlas(version=version)
            assert atlas.version == f"v{version}"
            assert len(atlas.technique_list) > 0
            assert len(atlas.mitigation_list) > 0
            assert len(atlas.casestudy_list) > 0

    def test_invalid_version_raises(self) -> None:
        with pytest.raises(ValueError, match="version must be one of"):
            Atlas(version="99.99.99")
