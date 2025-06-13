"""
ATLASを表現するクラス
現在ケーススタディーについては使用予定がないため作成していない
もし使用する場合は別途yamlファイルからcasestudyの構成を取得する必要あり(関数についてはwork1のdatabase.pyを参照)
"""

import os
import re
from collections.abc import Iterator, Sequence
from contextlib import suppress
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypeVar

import chromadb
import polars as pl
import yaml
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI
from platformdirs import user_data_dir

T = TypeVar("T")


class AtlasTechnique:  # 1つのテクニックを表すクラス
    def __init__(  # noqa: PLR0913 (引数総数警告)
        self,
        name: str,
        technique_id: str,
        description: str,
        *,
        snake_case_name: str | None = None,
        have_parent: bool = False,
        parent_id: str | None = None,
        vector: list[float] | None = None,
        tactics: list["AtlasTactic"] | None = None,
    ) -> None:
        self.name = name
        self.id = technique_id
        self.description = self.clean_description(description)  # リンク系統を清掃する
        self.have_parent = have_parent
        self.parent_id = parent_id
        self.description_vector = vector  # ベクトル化されたdesc
        self.tactics = tactics  # tacticオブジェクト
        self.snake_case_name = snake_case_name

    def clean_description(self, desc: str) -> str:  # リンクなどのノイズのみを削除する関数
        pattern = r"\[(.*?)\]\(.*?\)"  # リンク表現(表示される部分のみをキャプチャ)
        replacement = r"\1"  # \1で最初のキャプチャグループ(表示される部分)のみに置き換え
        result = re.sub(pattern, replacement, desc)
        return result


class AtlasTactic:
    def __init__(self, tactic_id: str, name: str, description: str, tec_lis: list[AtlasTechnique], *, snake_case_name: str | None = None) -> None:
        self.id = tactic_id
        self.name = name
        self.snake_case_name = snake_case_name
        self.description = description
        self.technique_list = tec_lis  # list[AtlasTechnique]の形を保持したリスト


class AtlasMitigation:  # 1つの緩和策を表すクラス
    def __init__(self, mitigation_id: str, description: str, tec_lis: list[AtlasTechnique], *, snake_case_name: str | None = None) -> None:
        self.id = mitigation_id
        self.description = self.clean_description(description)  # リンク系統を清掃する
        self.technique_list = tec_lis
        self.snake_case_name = snake_case_name

    def check_technique_by_id(self, technique_id: str) -> bool:  # ある緩和策内にテクニックが含まれるかを確かめる関数
        return any(tec.id == technique_id for tec in self.technique_list)

    def clean_description(self, desc: str) -> str:
        # リンクなどのノイズのみを削除する関数(mitigationには現在はリンクなどはないが今後に備えてテクニックと同等の関数を用意する)  # noqa: ERA001
        pattern = r"\[(.*?)\]\(.*?\)"  # リンク表現(表示される部分のみをキャプチャ)
        replacement = r"\1"  # \1で最初のキャプチャグループ(表示される部分)のみに置き換え
        result = re.sub(pattern, replacement, desc)
        return result


class AtlasCaseStudyStep:
    def __init__(self, casestudy_step_id: str, tactic: AtlasTactic, technique: AtlasTechnique, description: str, parent_id: str) -> None:
        self.id = casestudy_step_id  # 親ケーススタディーのID + ステップ番号(AML.CS0000.0の形式で記述)
        self.tactic = tactic
        self.technique = technique
        self.description = description
        self.parent_id = parent_id  # ケーススタディーの親ID(AML.CS0000の形式で記述)


class AtlasCaseStudy:  # 1つのケーススタディを表すクラス
    def __init__(  # noqa: PLR0913 引数総数警告
        self,
        casestudy_id: str,
        name: str,
        summary: str,
        step_list: list[AtlasCaseStudyStep],
        target: str,
        acttor: str,
        casestudy_type: Literal["exercise", "incident"],
        *,
        reference_title_list: list[str] | None = None,
        reference_url_list: list[str] | None = None,
    ) -> None:
        self.id = casestudy_id
        self.name = name
        self.summary = summary
        self.procedure = step_list
        self.target = target
        self.actor = acttor
        self.type = casestudy_type
        self.reference_title_list = reference_title_list
        self.reference_url_list = reference_url_list


class Atlas:  # Atlasの機能を保持したクラス
    def __init__(
        self,
        *,  # 以下をキーワード引数に
        version: Literal["4.8.0", "4.9.0"] = "4.9.0",
        emb_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-large",
        initialize_vector: bool = False,
    ) -> None:
        self.version = f"v{version}"
        self.data_dir_path = files("atlas.data").joinpath(f"versions/{self.version}")  # パッケージ内のdataディレクトリ
        self.user_data_dir_path = Path(user_data_dir("atlas")) / "versions" / self.version  # ユーザ側dataディレクトリ
        self.user_data_dir_path.mkdir(parents=True, exist_ok=True)  # 念の為作成
        self.__create_tactic_list()
        self.__create_tec_list()
        self.__create_mit_list()
        self.__create_casestudy_list()
        self.__clean_description()  # 全ての記述内部に埋め込まれているリンクを削除
        if not os.path.isdir(str(self.user_data_dir_path.joinpath("chroma"))):  # ユーザ側のディレクトリが存在しない場合
            print("ベクトルDBの設定がありません。初期化し作成します...")
            initialize_vector = True  # 初期実行時なので初期化を行う
        self.chroma_client = chromadb.PersistentClient(str(self.user_data_dir_path.joinpath("chroma")))
        if initialize_vector or not os.path.isdir(self.user_data_dir_path.joinpath("chroma")):
            # 初期化が選択されている or 指定バージョンのvector DBが存在しない場合
            self.__initialize_vector(model=emb_model)
            self.__create_tec_list()  # ベクトルを新しい物に置き換えて再実行
            self.__create_mit_list()  # ベクトルを新しい物に置き換えて再実行
            self.__clean_description()  # 全ての記述内部に埋め込まれているリンクを削除(作り直してしまうためもう一度)
        self.technique_chroma_collection = self.__get_technique_chroma_collection(model=emb_model)
        self.casestudy_chroma_collection = self.__get_casestudy_chroma_collection(model=emb_model)

    def __clean_description(self) -> None:
        for tac in self.tactic_list:
            tac.description = self.__clean_one_description(tac.description)
        for tec in self.technique_list:
            tec.description = self.__clean_one_description(tec.description)
        for mit in self.mitigation_list:
            mit.description = self.__clean_one_description(mit.description)
        for cs in self.casestudy_list:
            cs.summary = self.__clean_one_description(cs.summary)

    def __clean_one_description(self, desc: str) -> str:  # 1つの記述についてリンク等を削除する関数
        def replace_snake_case_with_name(match: re.Match) -> str:
            snake_case_name = match.group(1)
            obj = self.__search_object_from_snake_case_name(snake_case_name)
            return f"{obj.name}({obj.id})"

        return re.sub(r"{{ create_internal_link\((.*?)\) }}", replace_snake_case_with_name, desc)

    def __search_object_from_snake_case_name(self, snake_case_name: str) -> AtlasTactic | AtlasTechnique | AtlasMitigation:
        for tactic in self.tactic_list:
            if tactic.snake_case_name == snake_case_name:
                return tactic
        for tec in self.technique_list:
            if tec.snake_case_name == snake_case_name:
                return tec
        for mit in self.mitigation_list:
            if mit.snake_case_name == snake_case_name:
                return mit
        err_msg = f"Object with snake_case_name '{snake_case_name}' not found."
        raise ValueError(err_msg)

    def __create_tactic_list(self) -> None:
        self.tactic_list: list[AtlasTactic] = []  # 一度初期化
        with open(self.data_dir_path.joinpath("yaml/tactics.yaml")) as f:
            yaml_data = yaml.safe_load(f)
        with open(self.data_dir_path.joinpath("yaml/tactics.yaml")) as f:
            text_data = f.read()
            anchor_list = re.findall(r"- &(.*)", text_data)
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
        if os.path.isfile(
            self.data_dir_path.joinpath("technique_vector.avro"),
        ):  # embeddingしたファイルがあるかどうかでtecにvectorを加えるかを決定
            use_vector = True
            vector_df = pl.read_avro(self.data_dir_path.joinpath("technique_vector.avro"))
        else:
            use_vector = False

        with open(self.data_dir_path.joinpath("yaml/techniques.yaml")) as f:
            yaml_data = yaml.safe_load(f)
        with open(self.data_dir_path.joinpath("yaml/techniques.yaml")) as f:
            text_data = f.read()
            anchor_list = re.findall(r"- &(.*)", text_data)
        self.technique_list: list[AtlasTechnique] = []  # 戻り値用のテクニックリスト

        for tec_dict, snake_case_name in zip(yaml_data, anchor_list, strict=False):
            if "subtechnique-of" in tec_dict:  # 子テクニックの場合
                parent_snake_case_name = re.findall(r"{{(.*)\.id}}", tec_dict["subtechnique-of"])[0]
                parent = self.__search_object_from_snake_case_name(snake_case_name=parent_snake_case_name)
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
                for tactic in parent.tactics:
                    tactic.technique_list.append(tec)  # tacticのテクニックリストに追加
            else:  # 親テクニックの場合
                tactics = [  # 各tacticを検索
                    self.__search_object_from_snake_case_name(snake_case_name=re.findall(r"{{(.*)\.id}}", tac)[0]) for tac in tec_dict["tactics"]
                ]
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
        with open(self.data_dir_path.joinpath("yaml/mitigations.yaml")) as f:
            yaml_data = yaml.safe_load(f)
        with open(self.data_dir_path.joinpath("yaml/techniques.yaml")) as f:
            text_data = f.read()
            anchor_list = re.findall(r"- &(.*)", text_data)
        self.mitigation_list: list[AtlasMitigation] = []
        for mit_dict, snake_case_name in zip(yaml_data, anchor_list, strict=False):
            tec_lis = [
                self.__search_object_from_snake_case_name(snake_case_name=re.findall(r"{{(.*)\.id}}", tec["id"])[0]) for tec in mit_dict["techniques"]
            ]
            mit = AtlasMitigation(
                mitigation_id=mit_dict["id"],
                description=mit_dict["description"],
                snake_case_name=snake_case_name,
                tec_lis=tec_lis,
            )
            self.mitigation_list.append(mit)

    def __create_casestudy_list(self) -> None:  # ケーススタディーの初期化+作成関数
        yaml_file_name_list = os.listdir(self.data_dir_path.joinpath("yaml/case-studies"))
        if ".DS_Store" in yaml_file_name_list:
            yaml_file_name_list.remove(".DS_Store")
        self.casestudy_list: list[AtlasCaseStudy] = []
        for yaml_file_name in yaml_file_name_list:
            with open(self.data_dir_path.joinpath(f"yaml/case-studies/{yaml_file_name}")) as f:
                yaml_data = yaml.safe_load(f)
            step_lis: list[AtlasCaseStudyStep] = []
            for i, step in enumerate(yaml_data["procedure"]):
                if re.search(r"AML\.TA\d{4}", step["tactic"]):  # AML.TA0000の形式で記述されている場合
                    tac_id = re.findall(r"(AML\..*)", step["tactic"])[0]
                    tac = self.search_tec_from_id(tec_id=tac_id)
                else:  # 通常のsnake_case + .id で記述されている場合
                    tac_snake_case_name = re.findall(r"{{(.*)\.id}}", step["tactic"])[0]
                    tac = self.__search_object_from_snake_case_name(snake_case_name=tac_snake_case_name)
                if re.search(r"AML\.T\d{4}", step["technique"]):  # AML.T0000の形式で記述されている場合
                    tec_id = re.findall(r"(AML\..*)", step["technique"])[0]
                    tec = self.search_tec_from_id(tec_id=tec_id)
                else:  # 通常のsnake_case + .id で記述されている場合
                    tec_snake_case_name = re.findall(r"{{(.*)\.id}}", step["technique"])[0]
                    tec = self.__search_object_from_snake_case_name(snake_case_name=tec_snake_case_name)
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
                acttor=yaml_data["actor"],
                casestudy_type=yaml_data["case-study-type"],
                reference_title_list=[d["title"] for d in yaml_data["references"]] if "references" in yaml_data else None,
                reference_url_list=[d["url"] for d in yaml_data["references"]] if "references" in yaml_data else None,
            )
            self.casestudy_list.append(casestudy)

    def __initialize_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
        self.__initialize_technique_vector(model=model)  # テクニックのベクトルDBを初期化
        self.__initialize_casestudy_vector(model=model)
        print("ベクトルDBの初期化が完了しました。")  # 初期化完了のメッセージ

    def __initialize_technique_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
        # ベクトルdb(chroma)とavroファイルの両方を初期化する関数
        # ベクトルavroファイルの初期化
        id_list = []
        desc_list = []
        metadata_list = []  # {"is_parent":bool}の形を保持した辞書
        for tec in self.technique_list:
            id_list.append(tec.id)
            desc_list.append(tec.description)
            metadata_list.append({"is_parent": not tec.have_parent})
        vector_list = self.__create_embedding_multiple(s_list=desc_list, model=model)
        vector_df = pl.DataFrame({"ID": id_list, "vector": vector_list})
        vector_df.write_avro(self.data_dir_path.joinpath("technique_vector.avro"))  # vector-DBを作成
        # ベクトルDB(chroma)の初期化
        with suppress(NotFoundError):
            self.chroma_client.delete_collection(name="atlas_technique")  # 存在する場合は一度削除してリセット
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(  # ベクトル化関数
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
        collection = self.chroma_client.get_or_create_collection(
            name="atlas_technique",
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef,
        )
        collection.add(documents=desc_list, ids=id_list, metadatas=metadata_list)  # コレクションに追加

    def __initialize_casestudy_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
        # ベクトルdb(chroma)とavroファイルの両方を初期化する関数
        # ベクトルavroファイルの初期化
        id_list = []
        desc_list = []
        metadata_list = []  # {"step_number":int}の形を保持した辞書

        for cs in self.casestudy_list:
            for i, step in enumerate(cs.procedure):
                id_list.append(step.id)  # AML.CS0000.0の形式
                desc_list.append(step.description)
                metadata_list.append({"step_number": i})

        vector_list = self.__create_embedding_multiple(s_list=desc_list, model=model)
        vector_df = pl.DataFrame({"ID": id_list, "vector": vector_list})
        vector_df.write_avro(self.data_dir_path.joinpath("casestudy_vector.avro"))  # vector-DBを作成
        # ベクトルDB(chroma)の初期化
        with suppress(NotFoundError):
            self.chroma_client.delete_collection(name="atlas_casestudy")  # 存在する場合は一度削除してリセット
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(  # ベクトル化関数
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
        collection = self.chroma_client.get_or_create_collection(
            name="atlas_casestudy",
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef,
        )
        collection.add(documents=desc_list, ids=id_list, metadatas=metadata_list)  # コレクションに追加

    def __get_technique_chroma_collection(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
    ) -> Collection:  # chromaDBを起動する関数
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
        collection = self.chroma_client.get_collection(name="atlas_technique", embedding_function=openai_ef)
        return collection

    def __get_casestudy_chroma_collection(
        self,
        model: Literal["text-embedding-3-small", "text-embedding-3-large"],
    ) -> Collection:  # chromaDBを起動する関数
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
        collection = self.chroma_client.get_collection(name="atlas_casestudy", embedding_function=openai_ef)
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
            fileter_parent (str): 親のみ, 子のみ, 両方 の3種類でフィルターをかける

        Returns:
            list[Atlas_Technique]: top_kで指定された個数分上位の結果をテクニックオブジェクト
        """
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
    ) -> list[AtlasCaseStudy]:
        """
        クエリを元にベクトルDBを検索する関数

        Args:
            query (str): 検索文言
            top_k (int): 上位何件を取得するか

        Returns:
            list[AtlasCaseStudy]: top_kで指定された個数分上位の結果をケーススタディーオブジェクト
        """
        result = self.casestudy_chroma_collection.query(query_texts=[query], n_results=top_k)
        ret: list[AtlasCaseStudyStep] = [self.search_cs_step_from_id(cs_step_id=cs_step_id) for cs_step_id in result["ids"][0]]
        return ret

    def __create_embedding_single(self, s: str, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> list[float]:
        """
        文字列1つのみを1つのベクトルに変換する関数

        Args:
            s (str): ベクトル化対象文字列
            model (str): ベクトル化モデル

        Returns:
            list[float]: ベクトル
        """
        client = OpenAI()
        response = client.embeddings.create(
            input=s,
            model=model,
        )
        return response.data[0].embedding

    def __create_embedding_multiple(self, s_list: list[str], model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> list[list[float]]:
        """
        複数の文字列を複数のベクトルに変換する関数(長さによって10個ずつに区切ってベクトル化)

        Args:
            s_list (list[str]): ベクトル化対象文字列のリスト
            model (str): ベクトル化モデル

        Returns:
            list[list[float]]: ベクトルのリスト
        """
        client = OpenAI()
        ret_list: list[list[float]] = []

        def chunked_iterable(iterable: Sequence[T], chunk_size: int) -> Iterator[Sequence[T]]:
            """指定されたサイズで iterable をチャンクに分割する"""
            for i in range(0, len(iterable), chunk_size):
                yield iterable[i : i + chunk_size]

        for chunk in chunked_iterable(s_list, 10):
            response = client.embeddings.create(
                input=chunk,
                model=model,
            )
            ret_list += [d.embedding for d in response.data]
        return ret_list


def main() -> None:  # テスト用関数
    load_dotenv(dotenv_path="/workspace/.env", override=True)
    atlas = Atlas(version="4.9.0", emb_model="text-embedding-3-large", initialize_vector=True)

    print("テクニック数:", len(atlas.technique_list))
    print("緩和策数:", len(atlas.mitigation_list))
    print("ケーススタディ数:", len(atlas.casestudy_list))
    print("タクティック数:", len(atlas.tactic_list))
    print("=================")

    print("テクニックのID:", atlas.technique_list[0].id)
    print("テクニックの名前:", atlas.technique_list[0].name)
    print("テクニックの説明:", atlas.technique_list[0].description)
    print("テクニックのタクティック", [tactic.name for tactic in atlas.technique_list[0].tactics])
    print("=================")

    print("緩和策のID:", atlas.mitigation_list[0].id)
    print("緩和策の説明:", atlas.mitigation_list[0].description)
    print("緩和策のテクニック", [tec.id for tec in atlas.mitigation_list[0].technique_list])
    print("=================")

    print("ケーススタディーのID", atlas.casestudy_list[0].id)
    print("ケーススタディーの名前", atlas.casestudy_list[0].name)
    print("ケーススタディーの説明", atlas.casestudy_list[0].description)
    print("ケーススタディーのステップ", [tec.id for tec in atlas.casestudy_list[0].technique_list])
    print("=================")

    test_query = "Please search techniques about LLM and RAG"
    searched_tec_lis = atlas.search_relevant_technique(query=test_query, top_k=5, filter="both")
    print("検索結果", [tec.id for tec in searched_tec_lis])

    for tec in atlas.technique_list:
        print("=================")
        print(tec.description)


if __name__ == "__main__":
    main()
