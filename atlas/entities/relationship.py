from typing import Any, Literal

from .reference import AtlasReference

RelationshipType = Literal[
    "achieves",
    "specializes",
    "mitigates",
    "employs",
    "sequences",
]


class AtlasRelationship:
    """
    v6形式の relationship (source と target を結ぶ typed edge) を表すクラス。

    v6.0.0時点では relationship 自体は uuid や references を持たないが、将来的な
    v6.x で first-class 化された場合に備えて Optional フィールドとして受け皿を用意する。

    Args:
        source_id (str): 起点エンティティID (例: "AML.CS0000")
        target_id (str): 対象エンティティID (例: "AML.T0000.001")
        type (RelationshipType): 関係種別 (achieves / specializes / mitigates / employs / sequences)
        description (str | None): 関係固有の説明 (mitigates / employs のみ)
        tactic_id (str | None): employs におけるステップのタクティックID
        step_id (str | None): employs におけるステップID (例: "S00")
        leads_to (list[str] | None): employs における次に遷移し得るstep-id一覧
        position (int | None): employs / sequences における順序
        uuid (str | None): 将来的にrelationshipへ付与される可能性のあるUUID
        references (list[AtlasReference] | None): 将来的にrelationshipへ付与される可能性のある参照
        raw (dict[str, Any]): 元のYAMLからロードした生辞書 (未モデル化フィールドの保持用)
    """

    def __init__(  # noqa: PLR0913
        self,
        source_id: str,
        target_id: str,
        type: RelationshipType,  # noqa: A002
        *,
        description: str | None = None,
        tactic_id: str | None = None,
        step_id: str | None = None,
        leads_to: list[str] | None = None,
        position: int | None = None,
        uuid: str | None = None,
        references: list[AtlasReference] | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self.source_id: str = source_id
        self.target_id: str = target_id
        self.type: RelationshipType = type
        self.description: str | None = description
        self.tactic_id: str | None = tactic_id
        self.step_id: str | None = step_id
        self.leads_to: list[str] | None = leads_to
        self.position: int | None = position
        self.uuid: str | None = uuid
        self.references: list[AtlasReference] | None = references
        self.raw: dict[str, Any] = raw if raw is not None else {}
