"""ATLASデータのバージョン(release / legacy format-version)を解決するモジュール。"""

from importlib.resources import files
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from importlib.abc import Traversable

_V6_FORMAT_VERSION = "6.0.0"


class ManifestEntry:
    """
    manifest.yamlの1リリースエントリを表すデータクラス。

    Args:
        release (str): リリース識別子 (例: "2026.06")
        release_date (str): リリース日付
        v6_path (str | None): このリリースにおけるv6形式ファイルの相対パス。無ければNone
        legacy_versions (list[str]): このリリースに含まれる旧format-version一覧
    """

    def __init__(
        self,
        release: str,
        release_date: str,
        v6_path: str | None,
        legacy_versions: list[str],
    ) -> None:
        self.release: str = release
        self.release_date: str = release_date
        self.v6_path: str | None = v6_path
        self.legacy_versions: list[str] = legacy_versions


def _load_manifest() -> list[ManifestEntry]:
    manifest_file: Traversable = files("atlas.data").joinpath("manifest.yaml")
    with manifest_file.open() as f:
        raw: list[dict] = yaml.safe_load(f)
    entries: list[ManifestEntry] = []
    for item in raw:
        v6_path: str | None = None
        legacy: list[str] = []
        for v in item.get("versions", []):
            fv = str(v["format-version"])
            if fv == _V6_FORMAT_VERSION:
                v6_path = v["path"]
            else:
                legacy.append(fv)
        entries.append(
            ManifestEntry(
                release=str(item["release"]),
                release_date=str(item["release-date"]),
                v6_path=v6_path,
                legacy_versions=legacy,
            ),
        )
    return entries


def load_manifest() -> list[ManifestEntry]:
    """
    manifest.yamlをロードしリリースエントリのリストを返す。

    Returns:
        list[ManifestEntry]: manifest.yamlに記載順(新→旧)のエントリ一覧
    """
    return _load_manifest()


def list_releases() -> list[str]:
    """
    v6形式ファイルを持つ利用可能なrelease識別子のリストを返す。

    Returns:
        list[str]: release識別子の昇順ソートリスト (例: ["2021.05", ..., "2026.06"])
    """
    releases: list[str] = [e.release for e in _load_manifest() if e.v6_path is not None]
    return sorted(releases)


def list_legacy_versions() -> list[str]:
    """
    manifestに登録されている旧format-versionの一覧(重複除去・昇順)を返す。

    Returns:
        list[str]: 例 ["2.0.0", ..., "5.6.0"]
    """
    seen: set[str] = set()
    for e in _load_manifest():
        for lv in e.legacy_versions:
            seen.add(lv)
    return sorted(seen)


def list_available_versions() -> list[str]:
    """
    ユーザがversion引数として指定可能な文字列の一覧(release + legacy)を返す。

    Returns:
        list[str]: releaseと旧format-versionを合わせた昇順ソートリスト
    """
    return sorted(set(list_releases()) | set(list_legacy_versions()))


def resolve(version: str) -> tuple[str, str]:
    """
    ユーザ指定のバージョン文字列を(release, v6ファイル相対パス)に解決する。

    v6リリース識別子(例: "2026.06")と旧format-version(例: "5.6.0")の両方を
    受け付け、旧指定は同一リリースに含まれるv6ファイルに自動フォールバックする。
    先頭の "v" は許容する。

    Args:
        version (str): 解決対象のバージョン文字列

    Returns:
        tuple[str, str]: (解決先のrelease識別子, v6ファイルのdata配下相対パス)

    Raises:
        ValueError: 該当するエントリが見つからない、または対応するv6ファイルが無い場合
    """
    key: str = version.lstrip("v")
    entries: list[ManifestEntry] = _load_manifest()
    for e in entries:
        if e.release == key and e.v6_path is not None:
            return e.release, e.v6_path
    for e in entries:
        if key in e.legacy_versions:
            if e.v6_path is None:
                err = f"legacy version '{version}' が属する release '{e.release}' にv6ファイルがありません。"
                raise ValueError(err)
            return e.release, e.v6_path
    available: list[str] = list_available_versions()
    err = f"version must be one of {available}. '{version}' is given."
    raise ValueError(err)


def latest_release() -> str:
    """
    manifest.yaml先頭(最新)のrelease識別子を返す。

    Returns:
        str: 最新release (例: "2026.06")
    """
    for e in _load_manifest():
        if e.v6_path is not None:
            return e.release
    err = "manifest.yaml にv6形式のリリースがありません。"
    raise ValueError(err)
