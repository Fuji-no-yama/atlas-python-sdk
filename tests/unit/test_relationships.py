"""AtlasRelationship のプロパティに関するテスト。"""

from atlas import Atlas
from atlas.entities import AtlasRelationship


class TestRelationshipList:
    """atlas.relationships が第一級のリストとして構築されていることを確認する。"""

    def test_relationships_populated(self, atlas_default: Atlas) -> None:
        assert isinstance(atlas_default.relationships, list)
        assert len(atlas_default.relationships) > 0
        for rel in atlas_default.relationships:
            assert isinstance(rel, AtlasRelationship)

    def test_relationship_types_are_known(self, atlas_default: Atlas) -> None:
        known = {"achieves", "specializes", "mitigates", "employs", "sequences"}
        for rel in atlas_default.relationships:
            assert rel.type in known

    def test_derived_view_consistency_achieves(self, atlas_default: Atlas) -> None:
        """achieves relationship の数と materialize された tactic.technique_list の総数が一致すること。"""
        achieves_count = sum(1 for r in atlas_default.relationships if r.type == "achieves")
        materialized = sum(len(tac.technique_list) for tac in atlas_default.tactic_list)
        assert achieves_count == materialized

    def test_derived_view_consistency_mitigates(self, atlas_default: Atlas) -> None:
        mitigates_count = sum(1 for r in atlas_default.relationships if r.type == "mitigates")
        materialized = sum(len(mit.technique_list) for mit in atlas_default.mitigation_list)
        assert mitigates_count == materialized

    def test_derived_view_consistency_employs(self, atlas_default: Atlas) -> None:
        employs_count = sum(1 for r in atlas_default.relationships if r.type == "employs")
        materialized = sum(len(cs.procedure) for cs in atlas_default.casestudy_list)
        assert employs_count == materialized


class TestGetRelationshipsHelper:
    """Atlas.get_relationships のフィルタが期待通り動くことを確認する。"""

    def test_filter_by_type(self, atlas_default: Atlas) -> None:
        seq = atlas_default.get_relationships(type="sequences")
        assert all(r.type == "sequences" for r in seq)
        assert len(seq) == sum(1 for r in atlas_default.relationships if r.type == "sequences")

    def test_filter_by_source(self, atlas_default: Atlas) -> None:
        first_mit = atlas_default.mitigation_list[0]
        rels = atlas_default.get_relationships(source_id=first_mit.id, type="mitigates")
        assert len(rels) == len(first_mit.technique_list)
        assert all(r.source_id == first_mit.id for r in rels)
        assert {r.target_id for r in rels} == {t.id for t in first_mit.technique_list}

    def test_filter_by_target(self, atlas_default: Atlas) -> None:
        first_tec = atlas_default.technique_list[0]
        rels = atlas_default.get_relationships(target_id=first_tec.id, type="achieves")
        assert all(r.target_id == first_tec.id and r.type == "achieves" for r in rels)
