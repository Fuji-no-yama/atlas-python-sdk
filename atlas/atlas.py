"""ATLASのv6形式データをロードするクラス。"""

import os
import shutil
from contextlib import suppress
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import chromadb
import polars as pl
import yaml
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions
from platformdirs import user_data_dir

from atlas import version_resolver
from atlas.config.utils import create_embedding_multiple
from atlas.entities import (
    AtlasCaseStudy,
    AtlasCaseStudyStep,
    AtlasMitigation,
    AtlasReference,
    AtlasTactic,
    AtlasTechnique,
)

if TYPE_CHECKING:
    from importlib.abc import Traversable

    from chromadb.api.client import ClientAPI

T = TypeVar("T")


class Atlas:
    """
    ATLASのv6データを保持しアクセスAPIを提供するクラス。

    Args:
        version (str | None): リリース識別子 (例: "2026.06") または
            旧format-version (例: "5.6.0")。Noneの場合はmanifest.yaml先頭の最新リリース。
            旧format-versionを指定した場合はmanifest経由で同一リリースのv6ファイルにフォールバックする。
        emb_model (str): ベクトル化に使用するOpenAIモデル
        initialize_vector (bool): ベクトルDBを初期化するかどうか
    """

    def __init__(
        self,
        *,
        version: str | None = None,
        emb_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-large",
        initialize_vector: bool = False,
    ) -> None:
        release_input: str = version if version is not None else version_resolver.latest_release()
        release, v6_path = version_resolver.resolve(release_input)
        self.release: str = release
        self.version: str = f"v{release}"
        self.data_file: Traversable = files("atlas.data").joinpath(v6_path)
        self.user_data_dir_path: Path = Path(user_data_dir("atlas")) / "releases" / release
        self.user_data_dir_path.mkdir(parents=True, exist_ok=True)

        with self.data_file.open() as f:
            self._raw: dict[str, Any] = yaml.safe_load(f)

        self.__create_tactic_list()
        self.__create_tec_list()
        self.__create_mit_list()
        self.__create_casestudy_list()
        self.__apply_relationships()

        self.technique_chroma_collection: Collection | None = None
        self.casestudy_chroma_collection: Collection | None = None

        openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
        if openai_api_key:
            if not os.path.isdir(str(self.user_data_dir_path.joinpath("chroma"))):
                print("ベクトルDBの設定がありません。初期化し作成します...")
                initialize_vector = True
            self.chroma_client: ClientAPI = chromadb.PersistentClient(str(self.user_data_dir_path.joinpath("chroma")))
            if initialize_vector or not os.path.isdir(self.user_data_dir_path.joinpath("chroma")):
                self.__initialize_vector(model=emb_model)
                self.__attach_technique_vectors()
            self.technique_chroma_collection = self.__get_technique_chroma_collection(model=emb_model)
            self.casestudy_chroma_collection = self.__get_casestudy_chroma_collection(model=emb_model)

    @staticmethod
    def _make_references(raw: list[dict[str, Any]] | None) -> list[AtlasReference] | None:
        if not raw:
            return None
        refs: list[AtlasReference] = []
        for r in raw:
            title = r.get("title")
            url = r.get("url")
            if title is None or url is None:
                continue
            refs.append(AtlasReference(title=str(title), url=str(url), ref_id=r.get("id")))
        return refs or None

    def __create_tactic_list(self) -> None:
        self.tactic_list: list[AtlasTactic] = []
        self._tactics_by_id: dict[str, AtlasTactic] = {}
        for tid, td in self._raw.get("tactics", {}).items():
            tac = AtlasTactic(
                tactic_id=tid,
                name=td["name"],
                description=td["description"],
                tec_lis=[],
                uuid=td.get("uuid"),
                references=self._make_references(td.get("references")),
            )
            self.tactic_list.append(tac)
            self._tactics_by_id[tid] = tac

    def __create_tec_list(self) -> None:
        self.technique_list: list[AtlasTechnique] = []
        self._techniques_by_id: dict[str, AtlasTechnique] = {}
        for tid, td in self._raw.get("techniques", {}).items():
            tec = AtlasTechnique(
                name=td["name"],
                technique_id=tid,
                description=td["description"],
                have_parent=False,
                parent_id=None,
                tactics=[],
                uuid=td.get("uuid"),
                references=self._make_references(td.get("references")),
                platforms=td.get("platforms"),
                maturity=td.get("maturity"),
            )
            self.technique_list.append(tec)
            self._techniques_by_id[tid] = tec

    def __create_mit_list(self) -> None:
        self.mitigation_list: list[AtlasMitigation] = []
        self._mitigations_by_id: dict[str, AtlasMitigation] = {}
        for mid, md in self._raw.get("mitigations", {}).items():
            mit = AtlasMitigation(
                mitigation_id=mid,
                name=md["name"],
                description=md["description"],
                tec_lis=[],
                uuid=md.get("uuid"),
                references=self._make_references(md.get("references")),
            )
            self.mitigation_list.append(mit)
            self._mitigations_by_id[mid] = mit

    def __create_casestudy_list(self) -> None:
        self.casestudy_list: list[AtlasCaseStudy] = []
        self._casestudies_by_id: dict[str, AtlasCaseStudy] = {}
        for cid, cd in self._raw.get("case-studies", {}).items():
            refs = self._make_references(cd.get("references"))
            cs_type = cd.get("type", "Exercise")
            cs = AtlasCaseStudy(
                casestudy_id=cid,
                name=cd["name"],
                summary=cd.get("description", ""),
                step_list=[],
                target=cd.get("target", ""),
                actor=cd.get("actor", ""),
                casestudy_type=cs_type,
                reference_title_list=[r.title for r in refs] if refs else None,
                reference_url_list=[r.url for r in refs] if refs else None,
                uuid=cd.get("uuid"),
                references=refs,
            )
            self.casestudy_list.append(cs)
            self._casestudies_by_id[cid] = cs

    def __apply_relationships(self) -> None:  # noqa: C901, PLR0912
        relationships: dict[str, dict[str, list[dict[str, Any]]]] = self._raw.get("relationships", {})

        for src_id, groups in relationships.items():
            for rel in groups.get("achieves", []):
                tec = self._techniques_by_id.get(rel["source"])
                tac = self._tactics_by_id.get(rel["target"])
                if tec is None or tac is None:
                    continue
                if tac not in tec.tactics:
                    tec.tactics.append(tac)
                if tec not in tac.technique_list:
                    tac.technique_list.append(tec)
            for rel in groups.get("mitigates", []):
                mit = self._mitigations_by_id.get(rel["source"])
                tec = self._techniques_by_id.get(rel["target"])
                if mit is None or tec is None:
                    continue
                if tec not in mit.technique_list:
                    mit.technique_list.append(tec)
            for rel in groups.get("employs", []):
                cs = self._casestudies_by_id.get(rel["source"])
                tec = self._techniques_by_id.get(rel["target"])
                tac = self._tactics_by_id.get(rel.get("tactic"))
                if cs is None or tec is None or tac is None:
                    continue
                step_id: str = rel.get("step-id", "")
                cs.procedure.append(
                    AtlasCaseStudyStep(
                        casestudy_step_id=f"{cs.id}.{step_id}",
                        tactic=tac,
                        technique=tec,
                        description=rel.get("description", ""),
                        parent_id=cs.id,
                        step_id=step_id or None,
                        leads_to=rel.get("leads-to"),
                    ),
                )
            _ = src_id  # relationshipsのキーは source と一致するため未使用

        for src_id, groups in relationships.items():
            for rel in groups.get("specializes", []):
                child = self._techniques_by_id.get(rel["source"])
                parent = self._techniques_by_id.get(rel["target"])
                if child is None or parent is None:
                    continue
                child.have_parent = True
                child.parent_id = parent.id
            _ = src_id

        # 旧v6リリース(specializes未導入)向けのフォールバック: ID命名規則 "AML.T####.###" からサブテクニックを推定
        for tec in self.technique_list:
            if tec.have_parent:
                continue
            parts: list[str] = tec.id.split(".")
            subtechnique_parts_count: int = 3
            if len(parts) >= subtechnique_parts_count and parts[0] == "AML" and parts[1].startswith("T"):
                parent_id: str = ".".join(parts[:2])
                parent_tec = self._techniques_by_id.get(parent_id)
                if parent_tec is not None:
                    tec.have_parent = True
                    tec.parent_id = parent_id

        for cs in self.casestudy_list:
            cs.procedure.sort(key=lambda s: s.step_id or "")

    def __attach_technique_vectors(self) -> None:
        avro_path: Path = self.user_data_dir_path.joinpath("technique_vector.avro")
        if not avro_path.is_file():
            return
        vector_df: pl.DataFrame = pl.read_avro(str(avro_path))
        for tec in self.technique_list:
            rows = vector_df.filter(pl.col("ID") == tec.id)
            if rows.height > 0:
                tec.description_vector = rows[0, "vector"].to_list()

    def __initialize_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
        try:
            self.__initialize_technique_vector(model=model)
            self.__initialize_casestudy_vector(model=model)
            print("ベクトルDBの初期化が完了しました。")
        except Exception:
            shutil.rmtree(str(self.user_data_dir_path.joinpath("chroma")))
            raise

    def __initialize_technique_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
        id_list: list[str] = []
        desc_list: list[str] = []
        metadata_list: list[dict[str, bool]] = []
        for tec in self.technique_list:
            id_list.append(tec.id)
            desc_list.append(tec.description)
            metadata_list.append({"is_parent": not tec.have_parent})
        vector_list: list[list[int | float]] = create_embedding_multiple(s_list=desc_list, model=model)
        vector_df = pl.DataFrame({"ID": id_list, "vector": vector_list})
        vector_df.write_avro(str(self.user_data_dir_path.joinpath("technique_vector.avro")))
        with suppress(NotFoundError):
            self.chroma_client.delete_collection(name="atlas_technique")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
        collection: Collection = self.chroma_client.get_or_create_collection(
            name="atlas_technique",
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef,  # ty:ignore[invalid-argument-type]
        )
        collection.add(documents=desc_list, ids=id_list, metadatas=metadata_list)  # ty:ignore[invalid-argument-type]

    def __initialize_casestudy_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
        id_list: list[str] = []
        desc_list: list[str] = []
        metadata_list: list[dict[str, int]] = []

        for cs in self.casestudy_list:
            for i, step in enumerate(cs.procedure):
                id_list.append(step.id)
                desc_list.append(step.description)
                metadata_list.append({"step_number": i})

        vector_list: list[list[int | float]] = create_embedding_multiple(s_list=desc_list, model=model)
        vector_df = pl.DataFrame({"ID": id_list, "vector": vector_list})
        vector_df.write_avro(str(self.user_data_dir_path.joinpath("casestudy_vector.avro")))
        with suppress(NotFoundError):
            self.chroma_client.delete_collection(name="atlas_casestudy")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
        collection: Collection = self.chroma_client.get_or_create_collection(
            name="atlas_casestudy",
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef,  # ty:ignore[invalid-argument-type]
        )
        collection.add(documents=desc_list, ids=id_list, metadatas=metadata_list)  # ty:ignore[invalid-argument-type]

    def __get_technique_chroma_collection(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
    ) -> Collection:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
        return self.chroma_client.get_collection(name="atlas_technique", embedding_function=openai_ef)  # ty:ignore[invalid-argument-type]

    def __get_casestudy_chroma_collection(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
    ) -> Collection:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
        return self.chroma_client.get_collection(name="atlas_casestudy", embedding_function=openai_ef)  # ty:ignore[invalid-argument-type]

    def search_tec_from_id(self, tec_id: str) -> AtlasTechnique:
        """
        IDからテクニックを検索する。

        Args:
            tec_id (str): テクニックのID (例: "AML.T0042")

        Returns:
            AtlasTechnique: 検索されたテクニックオブジェクト
        """
        for tec in self.technique_list:
            if tec.id == tec_id:
                return tec
        err_msg = f"'{tec_id}'が検索の結果見つかりませんでした。"
        raise ValueError(err_msg)

    def search_tac_from_id(self, tac_id: str) -> AtlasTactic:
        """
        IDからタクティックを検索する。

        Args:
            tac_id (str): タクティックのID (例: "AML.TA0001")

        Returns:
            AtlasTactic: 検索されたタクティックオブジェクト
        """
        for tac in self.tactic_list:
            if tac.id == tac_id:
                return tac
        err_msg = f"'{tac_id}'が検索の結果見つかりませんでした。"
        raise ValueError(err_msg)

    def search_mit_from_id(self, mit_id: str) -> AtlasMitigation:
        """
        IDから緩和策を検索する。

        Args:
            mit_id (str): 緩和策のID (例: "AML.M0000")

        Returns:
            AtlasMitigation: 検索された緩和策オブジェクト
        """
        for mit in self.mitigation_list:
            if mit.id == mit_id:
                return mit
        err_msg = f"'{mit_id}'が検索の結果見つかりませんでした。"
        raise ValueError(err_msg)

    def search_cs_from_id(self, cs_id: str) -> AtlasCaseStudy:
        """
        IDからケーススタディを検索する。

        Args:
            cs_id (str): ケーススタディのID (例: "AML.CS0026")

        Returns:
            AtlasCaseStudy: 検索されたケーススタディオブジェクト
        """
        for cs in self.casestudy_list:
            if cs.id == cs_id:
                return cs
        err_msg = f"'{cs_id}'が検索の結果見つかりませんでした。"
        raise ValueError(err_msg)

    def search_cs_step_from_id(self, cs_step_id: str) -> AtlasCaseStudyStep:
        """
        ケーススタディIDとステップIDからステップを検索する。

        v6形式の "AML.CS0000.S00" と旧形式の "AML.CS0000.0" の両方を受け付ける。

        Args:
            cs_step_id (str): ケーススタディID+ステップID

        Returns:
            AtlasCaseStudyStep: 検索されたステップオブジェクト
        """
        for cs in self.casestudy_list:
            for step in cs.procedure:
                if step.id == cs_step_id:
                    return step
        err_msg = f"'{cs_step_id}'が検索の結果見つかりませんでした。"
        raise ValueError(err_msg)

    def search_relevant_technique(
        self,
        query: str,
        top_k: int,
        *,
        filter: Literal["parent", "child", "both"] = "both",  # noqa: A002
    ) -> list[AtlasTechnique]:
        """
        クエリを元にベクトルDBを検索し関連テクニックを返す。

        Args:
            query (str): 検索文言
            top_k (int): 上位取得件数
            filter (str): "parent"のみ / "child"のみ / "both"

        Returns:
            list[AtlasTechnique]: 関連テクニックのリスト
        """
        if self.technique_chroma_collection is None:
            err_msg = "ベクトルDB検索にはOPENAI_API_KEYの設定が必要です。"
            raise ValueError(err_msg)
        if filter == "parent":
            result = self.technique_chroma_collection.query(query_texts=[query], n_results=top_k, where={"is_parent": True})
        elif filter == "child":
            result = self.technique_chroma_collection.query(query_texts=[query], n_results=top_k, where={"is_parent": False})
        else:
            result = self.technique_chroma_collection.query(query_texts=[query], n_results=top_k)
        return [self.search_tec_from_id(tec_id=tec_id) for tec_id in result["ids"][0]]

    def search_relevant_casestudy(
        self,
        query: str,
        top_k: int,
    ) -> list[AtlasCaseStudyStep]:
        """
        クエリを元にベクトルDBを検索し関連するケーススタディステップを返す。

        Args:
            query (str): 検索文言
            top_k (int): 上位取得件数

        Returns:
            list[AtlasCaseStudyStep]: 関連するステップのリスト
        """
        if self.casestudy_chroma_collection is None:
            err_msg = "ベクトルDB検索にはOPENAI_API_KEYの設定が必要です。"
            raise ValueError(err_msg)
        result = self.casestudy_chroma_collection.query(query_texts=[query], n_results=top_k)
        return [self.search_cs_step_from_id(cs_step_id=cs_step_id) for cs_step_id in result["ids"][0]]

    def get_available_versions(self) -> list[str]:
        """
        利用可能なATLASリリース一覧を返す(v6形式のrelease識別子のみ)。

        Returns:
            list[str]: 昇順ソートされたrelease識別子のリスト
        """
        return version_resolver.list_releases()

    def get_available_legacy_versions(self) -> list[str]:
        """
        version引数に指定可能な旧format-version(legacy)一覧を返す。

        Returns:
            list[str]: 昇順ソートされた旧format-versionのリスト
        """
        return version_resolver.list_legacy_versions()
