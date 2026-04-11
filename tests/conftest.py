import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    """OPENAI_API_KEYが未設定の場合、integrationマーカー付きテストを自動スキップする。"""
    if os.environ.get("OPENAI_API_KEY"):
        return
    skip_marker = pytest.mark.skip(reason="OPENAI_API_KEY not set")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)
