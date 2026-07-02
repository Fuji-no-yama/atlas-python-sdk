import re
from typing import TYPE_CHECKING

from .reference import AtlasReference

if TYPE_CHECKING:
    from .tactic import AtlasTactic


class AtlasTechnique:
    """
    ATLASのテクニックを表すクラス。

    Args:
        name (str): 表示名
        technique_id (str): テクニックのID (例: "AML.T0000")
        description (str): 説明文(初期化時にリンクなどを清掃)
        have_parent (bool): サブテクニックかどうか
        parent_id (str | None): 親テクニックのID
        vector (list[float] | None): descriptionのembeddingベクトル
        tactics (list[AtlasTactic]): 属するタクティックのリスト
        uuid (str | None): v6形式で付与されるUUID
        references (list[AtlasReference] | None): 外部参照リスト
        platforms (list[str] | None): v6で追加されたプラットフォーム一覧
            (例: ["Predictive AI", "Generative AI", "Agentic AI", "Enterprise"])
        maturity (str | None): v6の成熟度 (例: "Demonstrated", "Feasible")
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        technique_id: str,
        description: str,
        *,
        have_parent: bool = False,
        parent_id: str | None = None,
        vector: list[float] | None = None,
        tactics: list["AtlasTactic"],
        uuid: str | None = None,
        references: list[AtlasReference] | None = None,
        platforms: list[str] | None = None,
        maturity: str | None = None,
    ) -> None:
        self.name: str = name
        self.id: str = technique_id
        self.description: str = self.clean_description(description)
        self.have_parent: bool = have_parent
        self.parent_id: str | None = parent_id
        self.description_vector: list[float] | None = vector
        self.tactics: list[AtlasTactic] = tactics
        self.uuid: str | None = uuid
        self.references: list[AtlasReference] | None = references
        self.platforms: list[str] | None = platforms
        self.maturity: str | None = maturity

    def clean_description(self, desc: str) -> str:
        """
        descriptionからマークダウンリンクの表示部分のみを残しURL部分を除去する。

        Args:
            desc (str): 元のdescription文字列

        Returns:
            str: 清掃後のdescription文字列
        """
        pattern = r"\[(.*?)\]\(.*?\)"
        replacement = r"\1"
        return re.sub(pattern, replacement, desc)
