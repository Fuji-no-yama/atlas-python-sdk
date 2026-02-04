"""
Atlasクラスの基本的なテスト
"""

import os

from atlas.config.settings import settings

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
        for x in os.environ.items():
            print(x)
        print("ATLAS_TEST_FLAG", settings.atlas_test_flag)
        assert settings.atlas_test_flag is True
