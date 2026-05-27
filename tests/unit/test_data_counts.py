"""データ件数が公式サイトの値と一致しているかを確認するテスト。"""

import pytest

from atlas.core import Atlas

# 公式サイト(https://atlas.mitre.org)から確認した各バージョンの期待値
# tactics: タクティック数
# techniques: 親テクニックのみの数
# sub_techniques: 子テクニックのみの数
# mitigations: 緩和策数
# casestudies: ケーススタディ数
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "5.0.0": {"tactics": 15, "techniques": 75, "sub_techniques": 55, "mitigations": 26, "casestudies": 33},
    "5.1.0": {"tactics": 16, "techniques": 84, "sub_techniques": 56, "mitigations": 32, "casestudies": 42},
    "5.2.0": {"tactics": 16, "techniques": 91, "sub_techniques": 56, "mitigations": 35, "casestudies": 45},
    "5.3.0": {"tactics": 16, "techniques": 92, "sub_techniques": 56, "mitigations": 35, "casestudies": 48},
    "5.4.0": {"tactics": 16, "techniques": 97, "sub_techniques": 58, "mitigations": 35, "casestudies": 52},
    "5.5.0": {"tactics": 16, "techniques": 101, "sub_techniques": 66, "mitigations": 35, "casestudies": 57},
    "5.6.0": {"tactics": 16, "techniques": 101, "sub_techniques": 69, "mitigations": 35, "casestudies": 57},
}


@pytest.fixture(scope="module", params=EXPECTED_COUNTS.keys())
def versioned_atlas(request: pytest.FixtureRequest) -> tuple[Atlas, dict[str, int]]:
    """バージョンごとのAtlasインスタンスと期待値のペアを返す。"""
    version: str = request.param
    atlas = Atlas(version=version)
    return atlas, EXPECTED_COUNTS[version]


class TestDataCounts:
    """各バージョンのデータ件数が公式サイトの値と一致しているかを確認する。"""

    def test_tactic_count(self, versioned_atlas: tuple[Atlas, dict[str, int]]) -> None:
        atlas, expected = versioned_atlas
        assert len(atlas.tactic_list) == expected["tactics"], f"{atlas.version}: tactic count mismatch"

    def test_technique_count(self, versioned_atlas: tuple[Atlas, dict[str, int]]) -> None:
        atlas, expected = versioned_atlas
        actual = sum(1 for tec in atlas.technique_list if not tec.have_parent)
        assert actual == expected["techniques"], f"{atlas.version}: technique (parent only) count mismatch"

    def test_sub_technique_count(self, versioned_atlas: tuple[Atlas, dict[str, int]]) -> None:
        atlas, expected = versioned_atlas
        actual = sum(1 for tec in atlas.technique_list if tec.have_parent)
        assert actual == expected["sub_techniques"], f"{atlas.version}: sub-technique count mismatch"

    def test_mitigation_count(self, versioned_atlas: tuple[Atlas, dict[str, int]]) -> None:
        atlas, expected = versioned_atlas
        assert len(atlas.mitigation_list) == expected["mitigations"], f"{atlas.version}: mitigation count mismatch"

    def test_casestudy_count(self, versioned_atlas: tuple[Atlas, dict[str, int]]) -> None:
        atlas, expected = versioned_atlas
        assert len(atlas.casestudy_list) == expected["casestudies"], f"{atlas.version}: casestudy count mismatch"
