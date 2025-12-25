import re

from .technique import AtlasTechnique


class AtlasMitigation:  # 1つの緩和策を表すクラス
    def __init__(self, mitigation_id: str, name: str, description: str, tec_lis: list[AtlasTechnique], *, snake_case_name: str | None = None) -> None:
        self.id: str = mitigation_id
        self.name: str = name
        self.description: str = self.clean_description(description)  # リンク系統を清掃する
        self.technique_list: list[AtlasTechnique] = tec_lis
        self.snake_case_name: str | None = snake_case_name

    def check_technique_by_id(self, technique_id: str) -> bool:  # ある緩和策内にテクニックが含まれるかを確かめる関数
        return any(tec.id == technique_id for tec in self.technique_list)

    def clean_description(self, desc: str) -> str:
        # リンクなどのノイズのみを削除する関数(mitigationには現在はリンクなどはないが今後に備えてテクニックと同等の関数を用意する)  # noqa: ERA001
        pattern = r"\[(.*?)\]\(.*?\)"  # リンク表現(表示される部分のみをキャプチャ)
        replacement = r"\1"  # \1で最初のキャプチャグループ(表示される部分)のみに置き換え
        result: str = re.sub(pattern, replacement, desc)
        return result
