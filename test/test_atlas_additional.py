"""
Atlasクラスの応用テスト(APIキーの使用)
"""

import os
import re

import pytest

from atlas.atlas import AtlasTechnique
from atlas.core import Atlas


class TestAtlas:
    def test_atlas_with_api_key(self) -> None:
        """
        全体のAtlasインスタンスをテストする
        """
        if os.environ["ATLAS_TEST_FLAG"] == "True":
            pytest.skip("APIキーを用いたテストはスキップされます。")
        atlas = Atlas(initialize_vector=True)  # 最新版のみについてテスト
        self.check_atlas(atlas)

    def check_atlas(self, atlas: Atlas) -> None:
        """
        Atlasインスタンスの基本プロパティを確認
        """
        self.check_properties(atlas)
        self.check_description(atlas)
        self.check_search(atlas)

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

    def check_search(self, atlas: Atlas) -> None:
        """
        検索機能が動作するかを確認する
        """
        n = 5
        query = "phishing LLM cloud"
        results = atlas.search_relevant_technique(query, top_k=n)
        assert results is not None
        assert isinstance(results, list)
        assert len(results) == n
        for tech in results:
            assert isinstance(tech, AtlasTechnique)
            assert tech.id is not None
            assert tech.name is not None
            assert tech.description is not None
