"""
ATLASを表現するクラス
現在ケーススタディーについては使用予定がないため作成していない
もし使用する場合は別途yamlファイルからcasestudyの構成を取得する必要あり(関数についてはwork1のdatabase.pyを参照)
"""

import os
import re
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

from atlas.config.settings import settings
from atlas.config.utils import create_embedding_multiple
from atlas.entities import AtlasCaseStudy, AtlasCaseStudyStep, AtlasMitigation, AtlasTactic, AtlasTechnique

if TYPE_CHECKING:
    from importlib.abc import Traversable

    from chromadb.api.client import ClientAPI
    from ty_extensions import Unknown

T = TypeVar("T")


class Atlas:  # Atlasの機能を保持したクラス
    def __init__(
        self,
        *,  # 以下をキーワード引数に
        version: str = "5.3.0",
        emb_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-large",
        initialize_vector: bool = False,
    ) -> None:
        """
        Args:
            version (str): ATLASデータバージョン ("4.4.0", "4.5.0", "4.6.0", "4.7.0", "4.8.0", "4.9.0", "5.0.0", "5.1.0", "5.2.0", "5.3.0"のいずれか) defaultは5.3.0
            emb_model (str): ベクトル化に使用するモデル
            initialize_vector (bool): ベクトルDBを初期化するかどうか(デフォルトはFalse。TrueにするとベクトルDBを再構築する)
        """  # noqa: E501
        available_versions: list[str] = self.get_available_versions()
        if version not in available_versions:
            err_msg = f"version must be one of {available_versions}. '{version}' is given."  # noqa: E501
            raise ValueError(err_msg)
        self.version = f"v{version}"
        self.data_dir_path: Traversable = files("atlas.data").joinpath(f"versions/{self.version}")  # パッケージ内のdataディレクトリ
        self.user_data_dir_path: Path = Path(user_data_dir("atlas")) / "versions" / self.version  # ユーザ側dataディレクトリ
        self.user_data_dir_path.mkdir(parents=True, exist_ok=True)  # 念の為作成
        self.__create_tactic_list()
        self.__create_tec_list()
        self.__create_mit_list()
        self.__create_casestudy_list()
        self.__clean_description()  # 全ての記述内部に埋め込まれているリンクを削除
        if not settings.atlas_test_flag:
            if settings.openai_api_key is None or settings.openai_api_key == "":
                err_msg = "OpenAI APIキーが設定されていません。環境変数'OPENAI_API_KEY'にAPIキーを設定してください。"
                raise ValueError(err_msg)
            if not os.path.isdir(str(self.user_data_dir_path.joinpath("chroma"))):  # ユーザ側のディレクトリが存在しない場合
                print("ベクトルDBの設定がありません。初期化し作成します...")
                initialize_vector = True  # 初期実行時なので初期化を行う
            self.chroma_client: ClientAPI = chromadb.PersistentClient(str(self.user_data_dir_path.joinpath("chroma")))
            if initialize_vector or not os.path.isdir(self.user_data_dir_path.joinpath("chroma")):
                # 初期化が選択されている or 指定バージョンのvector DBが存在しない場合
                self.__initialize_vector(model=emb_model)
                self.__create_tec_list()  # ベクトルを新しい物に置き換えて再実行
                self.__create_mit_list()  # ベクトルを新しい物に置き換えて再実行
                self.__clean_description()  # 全ての記述内部に埋め込まれているリンクを削除(作り直してしまうためもう一度)
            self.technique_chroma_collection: Collection = self.__get_technique_chroma_collection(model=emb_model)
            self.casestudy_chroma_collection: Collection = self.__get_casestudy_chroma_collection(model=emb_model)
        else:
            print("Atlas is initialized in test mode. Vector DB functionalities are disabled.")

    def __clean_description(self) -> None:
        for tac in self.tactic_list:
            tac.description: str = self.__clean_one_description(tac.description)
        for tec in self.technique_list:
            tec.description: str = self.__clean_one_description(tec.description)
        for mit in self.mitigation_list:
            mit.description: str = self.__clean_one_description(mit.description)
        for cs in self.casestudy_list:
            cs.summary: str = self.__clean_one_description(cs.summary)

    def __clean_one_description(self, desc: str) -> str:  # 1つの記述についてリンク等を削除する関数
        def replace_snake_case_with_name(match: re.Match) -> str:
            snake_case_name: Unknown = match.group(1)
            obj: AtlasTactic | AtlasTechnique | AtlasMitigation = self.__search_object_from_snake_case_name(snake_case_name)
            return f"{obj.name}({obj.id})"

        s: str = re.sub(r"{{ create_internal_link\((.*?)\) }}", replace_snake_case_with_name, desc)  # v4.6.0以降はこちらで削除
        return re.sub(r"{{(.*?)\.name}}", replace_snake_case_with_name, s)  # v4.4.0~v4.5.0はこちらで削除

    def __search_object_from_snake_case_name(self, snake_case_name: str) -> AtlasTactic | AtlasTechnique | AtlasMitigation:
        if hasattr(self, "tactic_list"):
            for tactic in self.tactic_list:
                if tactic.snake_case_name == snake_case_name:
                    return tactic
        if hasattr(self, "technique_list"):
            for tec in self.technique_list:
                if tec.snake_case_name == snake_case_name:
                    return tec
        if hasattr(self, "mitigation_list"):
            for mit in self.mitigation_list:
                if mit.snake_case_name == snake_case_name:
                    return mit
        err_msg = f"Object with snake_case_name '{snake_case_name}' not found. "
        err_msg += f"available or not -> tactic_list:{hasattr(self, 'tactic_list')}, technique_list: {hasattr(self, 'technique_list')}, mitigation_list: {hasattr(self, 'mitigation_list')}"  # noqa: E501
        raise ValueError(err_msg)

    def __create_tactic_list(self) -> None:
        self.tactic_list: list[AtlasTactic] = []  # 一度初期化
        with self.data_dir_path.joinpath("yaml/tactics.yaml").open() as f:
            yaml_data: Unknown = yaml.safe_load(f)
        with self.data_dir_path.joinpath("yaml/tactics.yaml").open() as f:
            text_data: str = f.read()
            anchor_list: list[Any] = re.findall(r"- &(.*)", text_data)
        for tactic_dict, snake_case_name in zip(yaml_data, anchor_list, strict=False):
            tactic = AtlasTactic(
                tactic_id=tactic_dict["id"],
                name=tactic_dict["name"],
                snake_case_name=snake_case_name,
                description=tactic_dict["description"],
                tec_lis=[],  # あとで格納
            )
            self.tactic_list.append(tactic)

    def __create_tec_list(self) -> None:  # テクニックのリストを初期化・作成する関数
        if self.data_dir_path.joinpath("technique_vector.avro").is_file():  # embeddingしたファイルがあるかどうかでtecにvectorを加えるかを決定
            use_vector = True
            vector_df: pl.DataFrame = pl.read_avro(str(self.data_dir_path.joinpath("technique_vector.avro")))
        else:
            use_vector = False

        with self.data_dir_path.joinpath("yaml/techniques.yaml").open() as f:
            yaml_data: Unknown = yaml.safe_load(f)
        with self.data_dir_path.joinpath("yaml/techniques.yaml").open() as f:
            text_data: str = f.read()
            anchor_list: list[Any] = re.findall(r"- &(.*)", text_data)
        self.technique_list: list[AtlasTechnique] = []  # 戻り値用のテクニックリスト

        for tec_dict, snake_case_name in zip(yaml_data, anchor_list, strict=False):
            if "subtechnique-of" in tec_dict:  # 子テクニックの場合
                parent_snake_case_name: Any = re.findall(r"{{(.*)\.id}}", tec_dict["subtechnique-of"])[0]
                parent: AtlasTactic | AtlasTechnique | AtlasMitigation = self.__search_object_from_snake_case_name(
                    snake_case_name=parent_snake_case_name,
                )
                if not isinstance(parent, AtlasTechnique):
                    err_msg = f"Parent of subtechnique '{tec_dict['name']}' must be AtlasTechnique. '{type(parent)}' is given."
                    raise TypeError(err_msg)
                tec = AtlasTechnique(
                    name=tec_dict["name"],
                    technique_id=tec_dict["id"],
                    description=tec_dict["description"],
                    snake_case_name=snake_case_name,
                    have_parent=True,
                    parent_id=parent.id,
                    vector=vector_df.filter(pl.col("ID") == tec_dict["id"])[0, "vector"].to_list() if use_vector else None,  # ベクトルを取得
                    tactics=parent.tactics,
                )
                if not isinstance(parent.tactics, list):
                    err_msg = f"Parent of subtechnique '{tec_dict['name']}' must have 'tactics' as list. '{type(parent.tactics)}' is given."
                    raise TypeError(err_msg)
                for tactic in parent.tactics:
                    tactic.technique_list.append(tec)  # tacticのテクニックリストに追加
            else:  # 親テクニックの場合
                tactics: list[AtlasTactic] = []
                for tac in tec_dict["tactics"]:
                    tac_obj: AtlasTactic | AtlasTechnique | AtlasMitigation = self.__search_object_from_snake_case_name(
                        snake_case_name=re.findall(r"{{(.*)\.id}}", tac)[0],
                    )
                    if not isinstance(tac_obj, AtlasTactic):
                        err_msg = f"Tactics of technique '{tec_dict['name']}' must be AtlasTactic. '{type(tac_obj)}' is given."
                        raise TypeError(err_msg)
                    tactics.append(tac_obj)

                tec = AtlasTechnique(
                    name=tec_dict["name"],
                    technique_id=tec_dict["id"],
                    description=tec_dict["description"],
                    snake_case_name=snake_case_name,
                    have_parent=False,
                    parent_id=None,
                    vector=vector_df.filter(pl.col("ID") == tec_dict["id"])[0, "vector"].to_list() if use_vector else None,  # ベクトルを取得
                    tactics=tactics,
                )
                for tactic in tactics:
                    tactic.technique_list.append(tec)  # tacticのテクニックリストに追加
            self.technique_list.append(tec)

    def __create_mit_list(self) -> None:  # Atlas用のmitigationリスト作成関数
        with self.data_dir_path.joinpath("yaml/mitigations.yaml").open() as f:
            yaml_data: Unknown = yaml.safe_load(f)
        with self.data_dir_path.joinpath("yaml/techniques.yaml").open() as f:
            text_data: str = f.read()
            anchor_list: list[Any] = re.findall(r"- &(.*)", text_data)
        self.mitigation_list: list[AtlasMitigation] = []
        for mit_dict, snake_case_name in zip(yaml_data, anchor_list, strict=False):
            tec_lis: list[AtlasTechnique] = []
            for tec in mit_dict["techniques"]:
                tec_obj: AtlasTactic | AtlasTechnique | AtlasMitigation = self.__search_object_from_snake_case_name(
                    snake_case_name=re.findall(r"{{(.*)\.id}}", tec["id"])[0],
                )
                if not isinstance(tec_obj, AtlasTechnique):
                    err_msg = f"Techniques of mitigation '{mit_dict['id']}' must be AtlasTechnique. '{type(tec_obj)}' is given."
                    raise TypeError(err_msg)
                tec_lis.append(tec_obj)
            mit = AtlasMitigation(
                mitigation_id=mit_dict["id"],
                name=mit_dict["name"],
                description=mit_dict["description"],
                snake_case_name=snake_case_name,
                tec_lis=tec_lis,
            )
            self.mitigation_list.append(mit)

    def __create_casestudy_list(self) -> None:  # ケーススタディーの初期化+作成関数
        yaml_file_name_list: list[str] = os.listdir(str(self.data_dir_path.joinpath("yaml/case-studies")))
        if ".DS_Store" in yaml_file_name_list:
            yaml_file_name_list.remove(".DS_Store")
        self.casestudy_list: list[AtlasCaseStudy] = []
        for yaml_file_name in yaml_file_name_list:
            with self.data_dir_path.joinpath(f"yaml/case-studies/{yaml_file_name}").open() as f:
                yaml_data: Unknown = yaml.safe_load(f)
            step_lis: list[AtlasCaseStudyStep] = []
            for i, step in enumerate(yaml_data["procedure"]):
                if re.search(r"AML\.TA\d{4}", step["tactic"]):  # AML.TA0000の形式で記述されている場合
                    tac_id: Any = re.findall(r"(AML\..*)", step["tactic"])[0]
                    tac: AtlasTactic = self.search_tac_from_id(tac_id=tac_id)
                else:  # 通常のsnake_case + .id で記述されている場合
                    tac_snake_case_name: Any = re.findall(r"{{(.*)\.id}}", step["tactic"])[0]
                    tac: AtlasTactic | AtlasTechnique | AtlasMitigation = self.__search_object_from_snake_case_name(
                        snake_case_name=tac_snake_case_name,
                    )
                if re.search(r"AML\.T\d{4}", step["technique"]):  # AML.T0000の形式で記述されている場合
                    tec_id: Any = re.findall(r"(AML\..*)", step["technique"])[0]
                    tec: AtlasTechnique = self.search_tec_from_id(tec_id=tec_id)
                else:  # 通常のsnake_case + .id で記述されている場合
                    tec_snake_case_name: Any = re.findall(r"{{(.*)\.id}}", step["technique"])[0]
                    tec: AtlasTactic | AtlasTechnique | AtlasMitigation = self.__search_object_from_snake_case_name(
                        snake_case_name=tec_snake_case_name,
                    )

                if not isinstance(tac, AtlasTactic) or not isinstance(tec, AtlasTechnique):
                    err_msg = f"Tactic of casestudy step '{yaml_data['id']}.{i}' must be AtlasTactic. '{type(tac)}' is given."
                    err_msg += f"OR Technique of casestudy step '{yaml_data['id']}.{i}' must be AtlasTechnique. '{type(tec)}' is given."
                    raise TypeError(err_msg)

                step_lis.append(
                    AtlasCaseStudyStep(
                        casestudy_step_id=f"{yaml_data['id']}.{i}",
                        tactic=tac,
                        technique=tec,
                        description=step["description"],
                        parent_id=yaml_data["id"],
                    ),
                )

            casestudy = AtlasCaseStudy(
                casestudy_id=yaml_data["id"],
                name=yaml_data["name"],
                summary=yaml_data["summary"],
                step_list=step_lis,
                target=yaml_data["target"],
                actor=yaml_data["actor"],
                casestudy_type=yaml_data["case-study-type"],
                reference_title_list=[d["title"] for d in yaml_data["references"]] if "references" in yaml_data else None,
                reference_url_list=[d["url"] for d in yaml_data["references"]] if "references" in yaml_data else None,
            )
            self.casestudy_list.append(casestudy)

    def __initialize_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
        try:
            self.__initialize_technique_vector(model=model)  # テクニックのベクトルDBを初期化
            self.__initialize_casestudy_vector(model=model)
            print("ベクトルDBの初期化が完了しました。")  # 初期化完了のメッセージ
        except Exception:
            shutil.rmtree(str(self.user_data_dir_path.joinpath("chroma")))  # 初期化失敗時は変に残らないように削除
            raise

    def __initialize_technique_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
        # ベクトルdb(chroma)とavroファイルの両方を初期化する関数
        # ベクトルavroファイルの初期化
        id_list: list[str] = []
        desc_list: list[str] = []
        metadata_list: list[dict[str, bool]] = []  # {"is_parent":bool}の形を保持した辞書
        for tec in self.technique_list:
            id_list.append(tec.id)
            desc_list.append(tec.description)
            metadata_list.append({"is_parent": not tec.have_parent})
        vector_list: list[list[int | float]] = create_embedding_multiple(s_list=desc_list, model=model)
        vector_df = pl.DataFrame({"ID": id_list, "vector": vector_list})
        vector_df.write_avro(str(self.data_dir_path.joinpath("technique_vector.avro")))  # vector-DBを作成
        # ベクトルDB(chroma)の初期化
        with suppress(NotFoundError):
            self.chroma_client.delete_collection(name="atlas_technique")  # 存在する場合は一度削除してリセット
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(  # ベクトル化関数
            api_key=settings.openai_api_key,
            model_name=model,
        )
        collection: Collection = self.chroma_client.get_or_create_collection(
            name="atlas_technique",
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef,  # ty:ignore[invalid-argument-type]
        )
        collection.add(documents=desc_list, ids=id_list, metadatas=metadata_list)  # コレクションに追加  # ty:ignore[invalid-argument-type]

    def __initialize_casestudy_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
        # ベクトルdb(chroma)とavroファイルの両方を初期化する関数
        # ベクトルavroファイルの初期化
        id_list: list[str] = []
        desc_list: list[str] = []
        metadata_list: list[dict[str, int]] = []  # {"step_number":int}の形を保持した辞書

        for cs in self.casestudy_list:
            for i, step in enumerate(cs.procedure):
                id_list.append(step.id)  # AML.CS0000.0の形式
                desc_list.append(step.description)
                metadata_list.append({"step_number": i})

        vector_list: list[list[int | float]] = create_embedding_multiple(s_list=desc_list, model=model)
        vector_df = pl.DataFrame({"ID": id_list, "vector": vector_list})
        vector_df.write_avro(str(self.data_dir_path.joinpath("casestudy_vector.avro")))  # vector-DBを作成
        # ベクトルDB(chroma)の初期化
        with suppress(NotFoundError):
            self.chroma_client.delete_collection(name="atlas_casestudy")  # 存在する場合は一度削除してリセット
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(  # ベクトル化関数
            api_key=settings.openai_api_key,
            model_name=model,
        )
        collection: Collection = self.chroma_client.get_or_create_collection(
            name="atlas_casestudy",
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef,  # ty:ignore[invalid-argument-type]
        )
        collection.add(documents=desc_list, ids=id_list, metadatas=metadata_list)  # コレクションに追加  # ty:ignore[invalid-argument-type]

    def __get_technique_chroma_collection(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
    ) -> Collection:  # chromaDBを起動する関数
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=model,
        )
        collection: Collection = self.chroma_client.get_collection(name="atlas_technique", embedding_function=openai_ef)  # ty:ignore[invalid-argument-type]
        return collection

    def __get_casestudy_chroma_collection(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
    ) -> Collection:  # chromaDBを起動する関数
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=model,
        )
        collection: Collection = self.chroma_client.get_collection(name="atlas_casestudy", embedding_function=openai_ef)  # ty:ignore[invalid-argument-type]
        return collection

    def search_tec_from_id(self, tec_id: str) -> AtlasTechnique:  # IDからテクニックを検索する関数
        """
        IDからテクニックを検索する関数

        Args:
            tec_id (str): テクニックのID 例:AML.T0042

        Returns:
            AtlasTechnique: 検索されたテクニックオブジェクト
        """
        for tec in self.technique_list:
            if tec.id == tec_id:
                return tec
        err_msg = f"'{tec_id}'が検索の結果見つかりませんでした。"
        raise ValueError(err_msg)  # IDが見つからなかった場合はエラーを返す

    def search_tac_from_id(self, tac_id: str) -> AtlasTactic:  # IDからテクニックを検索する関数
        """
        IDからタクティクスを検索する関数

        Args:
            tac_id (str): タクティクスのID 例:AML.TA0001

        Returns:
            AtlasTactic: 検索されたタクティクスオブジェクト
        """
        for tac in self.tactic_list:
            if tac.id == tac_id:
                return tac
        err_msg = f"'{tac_id}'が検索の結果見つかりませんでした。"
        raise ValueError(err_msg)  # IDが見つからなかった場合はエラーを返す

    def search_mit_from_id(self, mit_id: str) -> AtlasMitigation:  # IDからテクニックを検索する関数
        """
        IDから緩和策を検索する関数

        Args:
            mit_id (str): テクニックのID 例:AML.M0000

        Returns:
            AtlasMitigation: 検索された緩和策オブジェクト
        """
        for mit in self.mitigation_list:
            if mit.id == mit_id:
                return mit
        err_msg = f"'{mit_id}'が検索の結果見つかりませんでした。"
        raise ValueError(err_msg)  # IDが見つからなかった場合はエラーを返す

    def search_cs_from_id(self, cs_id: str) -> AtlasCaseStudy:  # IDからテクニックを検索する関数
        """
        IDからケーススタディーを検索する関数

        Args:
            cs_id (str): ケーススタディーのID 例:AML.CS0026

        Returns:
            AtlasCaseStudy: 検索されたケーススタディーオブジェクト
        """
        for cs in self.casestudy_list:
            if cs.id == cs_id:
                return cs
        err_msg = f"'{cs_id}'が検索の結果見つかりませんでした。"
        raise ValueError(err_msg)  # IDが見つからなかった場合はエラーを返す

    def search_cs_step_from_id(self, cs_step_id: str) -> AtlasCaseStudyStep:
        """
        ケーススタディーのIDとステップ番号からケーススタディーのステップを検索する関数

        Args:
            cs_step_id (str): ケーススタディーのID + ステップ番号 例:AML.CS0026.0

        Returns:
            AtlasCaseStudyStep: 検索されたケーススタディーのステップオブジェクト
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
        クエリを元にベクトルDBを検索する関数

        Args:
            query (str): 検索文言
            top_k (int): 上位何件を取得するか
            filter (str): 親のみ, 子のみ, 両方 の3種類でフィルターをかける

        Returns:
            list[Atlas_Technique]: top_kで指定された個数分上位の結果をテクニックオブジェクト
        """
        if settings.atlas_test_flag:
            err_msg = "テストモードのため、ベクトルDB検索は無効化されています。"
            raise ValueError(err_msg)
        if filter == "parent":
            result = self.technique_chroma_collection.query(query_texts=[query], n_results=top_k, where={"is_parent": True})
        elif filter == "child":
            result = self.technique_chroma_collection.query(query_texts=[query], n_results=top_k, where={"is_parent": False})
        elif filter == "both":
            result = self.technique_chroma_collection.query(query_texts=[query], n_results=top_k)
        ret: list[AtlasTechnique] = [self.search_tec_from_id(tec_id=tec_id) for tec_id in result["ids"][0]]
        return ret

    def search_relevant_casestudy(
        self,
        query: str,
        top_k: int,
    ) -> list[AtlasCaseStudyStep]:
        """
        クエリを元にベクトルDBを検索し、関連するケーススタディーのステップを返す関数

        Args:
            query (str): 検索文言
            top_k (int): 上位何件を取得するか

        Returns:
            list[AtlasCaseStudyStep]: top_kで指定された個数分上位の結果をケーススタディーのステップオブジェクト
        """
        if settings.atlas_test_flag:
            err_msg = "テストモードのため、ベクトルDB検索は無効化されています。"
            raise ValueError(err_msg)
        result = self.casestudy_chroma_collection.query(query_texts=[query], n_results=top_k)
        ret: list[AtlasCaseStudyStep] = [self.search_cs_step_from_id(cs_step_id=cs_step_id) for cs_step_id in result["ids"][0]]
        return ret

    def get_available_versions(self) -> list[str]:
        """
        利用可能なATLASのバージョン一覧を取得する関数

        Returns:
            list[str]: 利用可能なATLASのバージョン一覧
        """
        versions: list[str] = []
        for p in files("atlas.data").joinpath("versions").iterdir():
            if re.match(r"v\d\.\d\.\d", p.name):
                version = p.name.lstrip("v")
                versions.append(version)
        return sorted(versions)
