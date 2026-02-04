from collections.abc import Iterator, Sequence
from typing import Literal, TypeVar

from openai import OpenAI

from .settings import settings

T = TypeVar("T")


def create_embedding_single(s: str, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> list[float]:
    """
    文字列1つのみを1つのベクトルに変換する関数

    Args:
        s (str): ベクトル化対象文字列
        model (str): ベクトル化モデル

    Returns:
        list[float]: ベクトル
    """
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        input=s,
        model=model,
    )
    return response.data[0].embedding


def create_embedding_multiple(s_list: list[str], model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> list[list[float]]:
    """
    複数の文字列を複数のベクトルに変換する関数(長さによって10個ずつに区切ってベクトル化)

    Args:
        s_list (list[str]): ベクトル化対象文字列のリスト
        model (Literal["text-embedding-3-small", "text-embedding-3-large"]): ベクトル化モデル

    Returns:
        list[list[float]]: ベクトルのリスト
    """
    client = OpenAI(api_key=settings.openai_api_key)
    ret_list: list[list[float]] = []

    def chunked_iterable(iterable: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
        """指定されたサイズで iterable をチャンクに分割する"""
        for i in range(0, len(iterable), chunk_size):
            yield iterable[i : i + chunk_size]

    for chunk in chunked_iterable(s_list, 10):
        response = client.embeddings.create(
            input=chunk,  # ty:ignore[invalid-argument-type]
            model=model,
        )
        ret_list += [d.embedding for d in response.data]
    return ret_list
