"""description文字列の清掃が正しく行われているかを確認するテスト。"""

import re

from atlas.core import Atlas


class TestDescriptionCleaning:
    """descriptionに未清掃のテンプレート表記が残っていないことを確認する。"""

    def _assert_clean(self, description: str) -> None:
        # 内部リンクのマークダウン表記 例: [{{name}}](/techniques/{{id}})
        pattern = r"\[\{\{.*?\}\}\]\(.*?\)"
        match = re.search(pattern, description)
        assert match is None, f"内部リンクのマークダウン表記が残っています: {match.group() if match else ''} in {description[:100]}"

        # テンプレート関数呼び出し 例: {{ create_internal_link(name) }}
        pattern = r"\{\{.*?\(.*?\)\}\}"
        match = re.search(pattern, description)
        assert match is None, f"テンプレート関数呼び出しが残っています: {match.group() if match else ''} in {description[:100]}"

    def test_technique_descriptions(self, atlas_default: Atlas) -> None:
        for tec in atlas_default.technique_list:
            self._assert_clean(tec.description)

    def test_mitigation_descriptions(self, atlas_default: Atlas) -> None:
        for mit in atlas_default.mitigation_list:
            self._assert_clean(mit.description)

    def test_casestudy_summaries(self, atlas_default: Atlas) -> None:
        for cs in atlas_default.casestudy_list:
            self._assert_clean(cs.summary)

    def test_casestudy_step_descriptions(self, atlas_default: Atlas) -> None:
        for cs in atlas_default.casestudy_list:
            for step in cs.procedure:
                self._assert_clean(step.description)


class TestDescriptionCleaningAllVersions:
    """全バージョンでdescription清掃が正しく行われていることを確認する。"""

    def _assert_clean(self, description: str) -> None:
        pattern = r"\[\{\{.*?\}\}\]\(.*?\)"
        assert re.search(pattern, description) is None, f"内部リンクが残っています: {description[:100]}"

        pattern = r"\{\{.*?\(.*?\)\}\}"
        assert re.search(pattern, description) is None, f"テンプレート関数が残っています: {description[:100]}"

    def test_all_versions(self, all_versions: list[str]) -> None:
        for version in all_versions:
            atlas = Atlas(version=version)
            for tec in atlas.technique_list:
                self._assert_clean(tec.description)
            for mit in atlas.mitigation_list:
                self._assert_clean(mit.description)
            for cs in atlas.casestudy_list:
                self._assert_clean(cs.summary)
                for step in cs.procedure:
                    self._assert_clean(step.description)
