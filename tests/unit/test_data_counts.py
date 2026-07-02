"""データ件数が公式サイトの値と一致しているかを確認するテスト。"""

import pytest

from atlas import Atlas

# v6形式リリースごとの期待値 (v6データより実測)
# tactics: タクティック数
# techniques: 親テクニックのみの数
# sub_techniques: 子テクニックのみの数
# mitigations: 緩和策数
# casestudies: ケーススタディ数
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "2025.09": {"tactics": 15, "techniques": 75, "sub_techniques": 55, "mitigations": 26, "casestudies": 33},
    "2025.10": {"tactics": 15, "techniques": 75, "sub_techniques": 55, "mitigations": 26, "casestudies": 33},
    "2025.11": {"tactics": 16, "techniques": 84, "sub_techniques": 56, "mitigations": 32, "casestudies": 42},
    "2025.11.2": {"tactics": 16, "techniques": 84, "sub_techniques": 56, "mitigations": 32, "casestudies": 42},
    "2025.12": {"tactics": 16, "techniques": 91, "sub_techniques": 56, "mitigations": 35, "casestudies": 45},
    "2026.01": {"tactics": 16, "techniques": 92, "sub_techniques": 56, "mitigations": 35, "casestudies": 48},
    "2026.02": {"tactics": 16, "techniques": 97, "sub_techniques": 58, "mitigations": 35, "casestudies": 52},
    "2026.03": {"tactics": 16, "techniques": 101, "sub_techniques": 66, "mitigations": 35, "casestudies": 57},
    "2026.04": {"tactics": 16, "techniques": 101, "sub_techniques": 69, "mitigations": 35, "casestudies": 57},
    "2026.05": {"tactics": 16, "techniques": 101, "sub_techniques": 69, "mitigations": 35, "casestudies": 57},
    "2026.06": {"tactics": 16, "techniques": 103, "sub_techniques": 70, "mitigations": 35, "casestudies": 63},
}

# 旧format-versionを指定した場合にどのreleaseにフォールバックされるかの期待値
LEGACY_FALLBACK: dict[str, str] = {
    "5.0.0": "2025.09",
    "5.0.1": "2025.10",
    "5.1.0": "2025.11",
    "5.1.1": "2025.11.2",
    "5.2.0": "2025.12",
    "5.3.0": "2026.01",
    "5.4.0": "2026.02",
    "5.5.0": "2026.03",
    "5.6.0": "2026.04",
}


@pytest.fixture(scope="module", params=EXPECTED_COUNTS.keys())
def versioned_atlas(request: pytest.FixtureRequest) -> tuple[Atlas, dict[str, int]]:
    """リリースごとのAtlasインスタンスと期待値のペアを返す。"""
    version: str = request.param
    atlas = Atlas(version=version)
    return atlas, EXPECTED_COUNTS[version]


class TestDataCounts:
    """各リリースのデータ件数が期待値と一致しているかを確認する。"""

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


class TestLegacyVersionFallback:
    """旧format-version指定が manifest 経由で正しい v6 リリースにフォールバックすることを確認する。"""

    @pytest.mark.parametrize(("legacy", "expected_release"), LEGACY_FALLBACK.items())
    def test_legacy_resolves_to_expected_release(self, legacy: str, expected_release: str) -> None:
        atlas = Atlas(version=legacy)
        assert atlas.release == expected_release, f"{legacy} should resolve to {expected_release}, got {atlas.release}"

    @pytest.mark.parametrize(("legacy", "expected_release"), LEGACY_FALLBACK.items())
    def test_legacy_counts_match_target_release(self, legacy: str, expected_release: str) -> None:
        atlas = Atlas(version=legacy)
        expected = EXPECTED_COUNTS[expected_release]
        assert len(atlas.tactic_list) == expected["tactics"]
        assert sum(1 for t in atlas.technique_list if not t.have_parent) == expected["techniques"]
        assert sum(1 for t in atlas.technique_list if t.have_parent) == expected["sub_techniques"]
        assert len(atlas.mitigation_list) == expected["mitigations"]
        assert len(atlas.casestudy_list) == expected["casestudies"]
