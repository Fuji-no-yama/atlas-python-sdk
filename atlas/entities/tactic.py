from .reference import AtlasReference
from .technique import AtlasTechnique


class AtlasTactic:
    """
    ATLASのタクティックを表すクラス。

    Args:
        tactic_id (str): タクティックのID (例: "AML.TA0000")
        name (str): 表示名
        description (str): 説明文
        tec_lis (list[AtlasTechnique]): このタクティックに紐づくテクニックのリスト
        uuid (str | None): v6形式で付与されるUUID。v5データからロードした場合はNone
        references (list[AtlasReference] | None): 外部参照リスト
    """

    def __init__(  # noqa: PLR0913
        self,
        tactic_id: str,
        name: str,
        description: str,
        tec_lis: list[AtlasTechnique],
        *,
        uuid: str | None = None,
        references: list[AtlasReference] | None = None,
    ) -> None:
        self.id: str = tactic_id
        self.name: str = name
        self.description: str = description
        self.technique_list: list[AtlasTechnique] = tec_lis
        self.uuid: str | None = uuid
        self.references: list[AtlasReference] | None = references
