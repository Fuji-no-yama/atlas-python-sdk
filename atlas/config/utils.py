from collections.abc import Callable
from typing import Any


def format_docstring(**kwargs: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    関数のdocstringを、指定されたキーワード引数でフォーマットするデコレーター。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # 元のdocstringが存在する場合のみ処理を行う
        if func.__doc__:
            # .format()メソッドでプレースホルダーを置換
            func.__doc__ = func.__doc__.format(**kwargs)
        return func

    return decorator
