import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tactic import AtlasTactic


class AtlasTechnique:  # 1つのテクニックを表すクラス
    def __init__(  # noqa: PLR0913 (引数総数警告)
        self,
        name: str,
        technique_id: str,
        description: str,
        *,
        snake_case_name: str | None = None,
        have_parent: bool = False,
        parent_id: str | None = None,
        vector: list[float] | None = None,
        tactics: list["AtlasTactic"],
    ) -> None:
        self.name: str = name
        self.id: str = technique_id
        self.description: str = self.clean_description(description)  # リンク系統を清掃する
        self.have_parent: bool = have_parent
        self.parent_id: str | None = parent_id
        self.description_vector: list[float] | None = vector  # ベクトル化されたdesc
        self.tactics: list[AtlasTactic] = tactics  # tacticオブジェクト
        self.snake_case_name: str | None = snake_case_name

    def clean_description(self, desc: str) -> str:  # リンクなどのノイズのみを削除する関数
        pattern = r"\[(.*?)\]\(.*?\)"  # リンク表現(表示される部分のみをキャプチャ)
        replacement = r"\1"  # \1で最初のキャプチャグループ(表示される部分)のみに置き換え
        result: str = re.sub(pattern, replacement, desc)
        return result
