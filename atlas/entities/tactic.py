from .technique import AtlasTechnique


class AtlasTactic:
    def __init__(self, tactic_id: str, name: str, description: str, tec_lis: list[AtlasTechnique], *, snake_case_name: str | None = None) -> None:
        self.id: str = tactic_id
        self.name: str = name
        self.snake_case_name: str | None = snake_case_name
        self.description: str = description
        self.technique_list: list[AtlasTechnique] = tec_lis  # list[AtlasTechnique]の形を保持したリスト
