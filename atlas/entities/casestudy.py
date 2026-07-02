from typing import Literal

from .reference import AtlasReference
from .tactic import AtlasTactic
from .technique import AtlasTechnique


class AtlasCaseStudyStep:
    """
    ケーススタディの1ステップ(procedure要素)を表すクラス。

    Args:
        casestudy_step_id (str): 親ID+step-idを結合したユニークID (例: "AML.CS0000.S00")
        tactic (AtlasTactic): このステップに対応するタクティック
        technique (AtlasTechnique): このステップに対応するテクニック
        description (str): 説明文
        parent_id (str): 親ケーススタディのID (例: "AML.CS0000")
        step_id (str | None): v6のstep-id(例: "S00")、旧形式からロード時はNone
        leads_to (list[str] | None): 次に遷移し得るstep-idのリスト。無ければNone
    """

    def __init__(  # noqa: PLR0913
        self,
        casestudy_step_id: str,
        tactic: AtlasTactic,
        technique: AtlasTechnique,
        description: str,
        parent_id: str,
        *,
        step_id: str | None = None,
        leads_to: list[str] | None = None,
    ) -> None:
        self.id: str = casestudy_step_id
        self.tactic: AtlasTactic = tactic
        self.technique: AtlasTechnique = technique
        self.description: str = description
        self.parent_id: str = parent_id
        self.step_id: str | None = step_id
        self.leads_to: list[str] | None = leads_to


class AtlasCaseStudy:
    """
    ATLASのケーススタディを表すクラス。

    Args:
        casestudy_id (str): ケーススタディID (例: "AML.CS0000")
        name (str): 表示名
        summary (str): 概要説明
        step_list (list[AtlasCaseStudyStep]): 手順ステップリスト
        target (str): 対象システム
        actor (str): 攻撃者
        casestudy_type (Literal["Exercise", "Incident"]): v6のcase-study type
        reference_title_list (list[str] | None): 参照タイトル一覧(下位互換用)
        reference_url_list (list[str] | None): 参照URL一覧(下位互換用)
        uuid (str | None): v6形式で付与されるUUID
        references (list[AtlasReference] | None): 構造化された参照リスト
    """

    def __init__(  # noqa: PLR0913
        self,
        casestudy_id: str,
        name: str,
        summary: str,
        step_list: list[AtlasCaseStudyStep],
        target: str,
        actor: str,
        casestudy_type: Literal["Exercise", "Incident"],
        *,
        reference_title_list: list[str] | None = None,
        reference_url_list: list[str] | None = None,
        uuid: str | None = None,
        references: list[AtlasReference] | None = None,
    ) -> None:
        self.id: str = casestudy_id
        self.name: str = name
        self.summary: str = summary
        self.procedure: list[AtlasCaseStudyStep] = step_list
        self.target: str = target
        self.actor: str = actor
        self.type: Literal["Exercise", "Incident"] = casestudy_type
        self.reference_title_list: list[str] | None = reference_title_list
        self.reference_url_list: list[str] | None = reference_url_list
        self.uuid: str | None = uuid
        self.references: list[AtlasReference] | None = references
