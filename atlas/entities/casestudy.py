from typing import Literal

from .tactic import AtlasTactic
from .technique import AtlasTechnique


class AtlasCaseStudyStep:
    def __init__(self, casestudy_step_id: str, tactic: AtlasTactic, technique: AtlasTechnique, description: str, parent_id: str) -> None:
        self.id: str = casestudy_step_id  # 親ケーススタディーのID + ステップ番号(AML.CS0000.0の形式で記述)
        self.tactic: AtlasTactic = tactic
        self.technique: AtlasTechnique = technique
        self.description: str = description
        self.parent_id: str = parent_id  # ケーススタディーの親ID(AML.CS0000の形式で記述)


class AtlasCaseStudy:  # 1つのケーススタディを表すクラス
    def __init__(  # noqa: PLR0913 引数総数警告
        self,
        casestudy_id: str,
        name: str,
        summary: str,
        step_list: list[AtlasCaseStudyStep],
        target: str,
        actor: str,
        casestudy_type: Literal["exercise", "incident"],
        *,
        reference_title_list: list[str] | None = None,
        reference_url_list: list[str] | None = None,
    ) -> None:
        self.id: str = casestudy_id
        self.name: str = name
        self.summary: str = summary
        self.procedure: list[AtlasCaseStudyStep] = step_list
        self.target: str = target
        self.actor: str = actor
        self.type: Literal["exercise", "incident"] = casestudy_type
        self.reference_title_list: list[str] | None = reference_title_list
        self.reference_url_list: list[str] | None = reference_url_list
