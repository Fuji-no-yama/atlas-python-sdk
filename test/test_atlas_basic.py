"""
Atlasクラスの基本的なテスト
"""

import re
from pathlib import Path

from atlas.core import Atlas

"""
メモ: チェックしたい内容
1. テクニック・緩和策・ケーススタディのリストが空でない
2. 各オブジェクト全てについて、諸々のプロパティが空もしくはNoneでない
3. descriptionにリンク表示などが含まれていない
"""


class TestAtlas:
    def test_atlas(self) -> None:
        """
        全体のAtlasインスタンスをテストする
        """
        # バージョンディレクトリを取得（絶対パスで）
        versions_dir = Path(__file__).parent.parent / "atlas" / "data" / "versions"
        available_versions: list[str] = [
            d.name[1:]  # "v5.2.0" -> "5.2.0"
            for d in versions_dir.iterdir()
            if d.is_dir() and d.name.startswith("v") and d.name[1:2].isdigit()
        ]

        for version in available_versions:
            atlas = Atlas(initialize_vector=True, version=version)  # type: ignore[arg-type]
            self.check_atlas(atlas)

    def check_atlas(self, atlas: Atlas) -> None:
        """
        Atlasインスタンスの基本プロパティを確認
        """
        self.check_properties(atlas)
        self.check_description(atlas)

    def check_properties(self, atlas: Atlas) -> None:
        """
        Atlasの基本プロパティを確認
        """
        assert atlas.version is not None
        assert atlas.version != ""
        assert atlas.technique_list is not None
        assert isinstance(atlas.technique_list, list)
        assert len(atlas.technique_list) > 0
        assert atlas.casestudy_list is not None
        assert isinstance(atlas.casestudy_list, list)
        assert len(atlas.casestudy_list) > 0
        assert atlas.mitigation_list is not None
        assert isinstance(atlas.mitigation_list, list)
        assert len(atlas.mitigation_list) > 0

    def check_description(self, atlas: Atlas) -> None:
        """
        descriptionのフィールドに清掃できていない文字列がないかを確認する
        """
        for tech in atlas.technique_list:
            self.check_one_description(tech.description)
        for mit in atlas.mitigation_list:
            self.check_one_description(mit.description)
        for cs in atlas.casestudy_list:
            self.check_one_description(cs.summary)
            for step in cs.procedure:
                self.check_one_description(step.description)

    def check_one_description(self, description: str) -> None:
        """
        1つのdescription文字列に清掃できていない文字列がないかを確認する
        - [paper by Le et al.](https://arxiv.org/abs/1802.03162)
        - [{{establish_accounts.name}}](/techniques/{{establish_accounts.id}})
        - {{ create_internal_link(victim_research_journals) }}
        """
        # 1つ目のパターン (現在は許容)
        # pattern = r"\[.*?\]\(https:.*?\)"
        # match = re.search(pattern, description)
        # assert match is None, f"外部リンクのマークダウン表記が残っています: {match.group() if match else ''} in {description[:100]}"

        # 2つ目のパターン
        pattern = r"\[\{\{.*?\}\}\]\(.*?\)"
        match = re.search(pattern, description)
        assert match is None, f"内部リンクのマークダウン表記が残っています: {match.group() if match else ''} in {description[:100]}"

        # 3つ目のパターン
        pattern = r"\{\{.*?\(.*?\)\}\}"
        match = re.search(pattern, description)
        assert match is None, f"テンプレート関数呼び出しのマークダウン表記が残っています: {match.group() if match else ''} in {description[:100]}"
