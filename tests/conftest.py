import os

import pytest
from dotenv import dotenv_values

_env = dotenv_values(".env.dev")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    """OPENAI_API_KEYが未設定の場合、integrationマーカー付きテストを自動スキップする。"""
    api_key = os.environ.get("OPENAI_API_KEY") or _env.get("OPENAI_API_KEY")
    if api_key:
        return
    skip_marker = pytest.mark.skip(reason="OPENAI_API_KEY not set")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)
