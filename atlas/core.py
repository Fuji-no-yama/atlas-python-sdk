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


class AtlasCaseStudy:  # 1つのケーススタディを表すクラス
    def __init__(  # noqa: PLR0913 引数総数警告
        self,
        casestudy_id: str,
        name: str,
        description: str,
        tec_list: list[AtlasTechnique],
        target: str,
        acttor: str,
        casestudy_type: Literal["exercise", "incident"],
        *,
        reference_title: str | None = None,
        reference_url: str | None = None,
    ) -> None:
        self.id = casestudy_id
        self.name = name
        self.description = description
        self.technique_list = tec_list
        self.target = target
        self.actor = acttor
        self.type = casestudy_type
        self.reference_title = reference_title
        self.reference_url = reference_url


class Atlas:  # Atlasの機能を保持したクラス
    def __init__(
        self,
        *,  # 以下をキーワード引数に
        version: Literal["4.8.0"] = "4.8.0",
        emb_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-large",
        initialize_vector: bool = False,
    ) -> None:
        self.version = f"v{version}"
        self.data_dir_path = files("atlas.data").joinpath(f"versions/{self.version}")  # パッケージ内のdataディレクトリ
        self.user_data_dir_path = Path(user_data_dir("atlas")) / "versions" / self.version  # ユーザ側dataディレクトリ
        self.user_data_dir_path.mkdir(parents=True, exist_ok=True)
        self.__create_tactic_list()
        self.__create_tec_list()
        self.__create_mit_list()
        self.__create_casestudy_list()
        self.chroma_client = chromadb.PersistentClient(str(self.user_data_dir_path.joinpath("chroma")))

        if initialize_vector or not os.path.isdir(self.user_data_dir_path.joinpath("chroma")):
            # 初期化が選択されている or 指定バージョンのvector DBが存在しない場合
            self.__initialize_vector(model=emb_model)
            self.__create_tec_list()  # ベクトルを新しい物に置き換えて再実行
            self.__create_mit_list()  # ベクトルを新しい物に置き換えて再実行
        self.chroma_collection = self.__get_chroma_collection(model=emb_model)

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

    def __create_tec_list_from_csv(self) -> list[AtlasTechnique]:  # テクニックのリストを作成する関数(Atlasクラスのinitで使用)
        if os.path.isfile(
            self.data_dir_path.joinpath("technique_vector.avro"),
        ):  # embeddingしたファイルがあるかどうかでtecにvectorを加えるかを決定
            use_vector = True
            vector_df = pl.read_avro(self.data_dir_path.joinpath("technique_vector.avro"))
        else:
            use_vector = False
        technique_df = pl.read_csv(self.data_dir_path.joinpath("atlas-techniques.csv"))
        ret_tec_list: list[AtlasTechnique] = []  # 戻り値用のテクニックリスト
        for row in technique_df.iter_rows(named=True):
            parent_id_list = re.findall(r"(AML\.T\d{4})\.", row["ID"])  # (AML.T~~).~~のカッコの部分を取得
            if len(parent_id_list) != 0:  # 親がいる場合(子の場合)
                have_parent = True
                parent_id = parent_id_list[0]
            else:  # 親がいない場合(親の場合)
                have_parent = False
                parent_id = None
            vector = vector_df.filter(pl.col("ID") == row["ID"])[0, "vector"].to_list() if use_vector else None  # ベクトルを取得
            tec = AtlasTechnique(
                name=row["name"],
                technique_id=row["ID"],
                description=row["description"],
                have_parent=have_parent,
                parent_id=parent_id,
                vector=vector,
                tactics=row["tactics"].split(", "),
            )
            ret_tec_list.append(tec)
        return ret_tec_list

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

    def __create_mit_list_from_csv(self) -> list[AtlasMitigation]:  # Atlas用のmitigationリスト作成関数
        mit_df = pl.read_csv(self.data_dir_path.joinpath("atlas-mitigations.csv"))
        mit_tec_df = pl.read_csv(self.data_dir_path.joinpath("atlas-mitigations-addressed-techniques.csv"))
        ret_mit_list: list[AtlasMitigation] = []
        for row in mit_df.iter_rows(named=True):
            tec_list: list[AtlasTechnique] = [  # dfに登録されているテクニック名から検索しリストに登録
                self.search_tec_from_id(tec_id=target_tec_id)
                for target_tec_id in mit_tec_df.filter(pl.col("source ID") == row["ID"])["target ID"].to_list()
            ]
            mit = AtlasMitigation(
                mitigation_id=row["ID"],
                description=row["description"],
                tec_lis=tec_list,
            )
            ret_mit_list.append(mit)
        return ret_mit_list

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
            tec_lis = []
            for step in yaml_data["procedure"]:
                if re.search(r"AML\.T\d{4}", step["technique"]):  # AML.T0000の形式で記述されている場合
                    tec_id = re.findall(r"(AML\..*)", step["technique"])[0]
                    tec_lis.append(self.search_tec_from_id(tec_id=tec_id))
                else:  # 通常のsnake_case + .id で記述されている場合
                    tec_snake_case_name = re.findall(r"{{(.*)\.id}}", step["technique"])[0]
                    tec_lis.append(self.__search_object_from_snake_case_name(snake_case_name=tec_snake_case_name))
            casestudy = AtlasCaseStudy(
                casestudy_id=yaml_data["id"],
                name=yaml_data["name"],
                description=yaml_data["summary"],
                tec_list=tec_lis,
                target=yaml_data["target"],
                acttor=yaml_data["actor"],
                casestudy_type=yaml_data["case-study-type"],
                reference_title=yaml_data["references"][0]["title"] if "references" in yaml_data else None,
                reference_url=yaml_data["references"][0]["url"] if "references" in yaml_data else None,
            )
            self.casestudy_list.append(casestudy)

    def __initialize_vector(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> None:
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

    def __get_chroma_collection(self, model: Literal["text-embedding-3-small", "text-embedding-3-large"]) -> Collection:  # chromaDBを起動する関数
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=model,
        )
        collection = self.chroma_client.get_collection(name="atlas_technique", embedding_function=openai_ef)
        return collection

    def search_tec_from_id(self, tec_id: str) -> AtlasTechnique:  # IDからテクニックを検索する関数
        """
        IDからテクニックを検索する関数

        Args:
            tec_id (str): テクニックのID 例:AML.T0042

        Returns:
            Atlas_Technique: 検索されたテクニックオブジェクト
        """
        for tec in self.technique_list:
            if tec.id == tec_id:
                return tec
        return

    def search_relevant_technique(
        self,
        query: str,
        top_k: int,
        *,
        filter_parent: Literal["parent", "child", "both"] = "both",
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
        if filter_parent == "parent":
            result = self.chroma_collection.query(query_texts=[query], n_results=top_k, where={"is_parent": True})
        elif filter_parent == "child":
            result = self.chroma_collection.query(query_texts=[query], n_results=top_k, where={"is_parent": False})
        elif filter_parent == "both":
            result = self.chroma_collection.query(query_texts=[query], n_results=top_k)
        ret: list[AtlasTechnique] = [self.search_tec_from_id(tec_id=tec_id) for tec_id in result["ids"][0]]
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
    load_dotenv(override=True)
    atlas = Atlas(version="4.8.0", emb_model="text-embedding-3-large", initialize_vector=True)
    print("テクニック数:", len(atlas.technique_list))
    print("緩和策数:", len(atlas.mitigation_list))
    print("ケーススタディ数:", len(atlas.casestudy_list))
    print("タクティック数:", len(atlas.tactic_list))

    print("テクニックのID:", atlas.technique_list[0].id)
    print("テクニックの名前:", atlas.technique_list[0].name)
    print("テクニックの説明:", atlas.technique_list[0].description)
    print("テクニックのタクティック", [tactic.name for tactic in atlas.technique_list[0].tactics])

    print("緩和策のID:", atlas.mitigation_list[0].id)
    print("緩和策の説明:", atlas.mitigation_list[0].description)
    print("緩和策のテクニック", [tec.id for tec in atlas.mitigation_list[0].technique_list])

    print("ケーススタディーのID", atlas.casestudy_list[0].id)
    print("ケーススタディーの名前", atlas.casestudy_list[0].name)
    print("ケーススタディーの説明", atlas.casestudy_list[0].description)
    print("ケーススタディーのステップ", [tec.id for tec in atlas.casestudy_list[0].technique_list])

    test_query = "Please search techniques about LLM and RAG"
    searched_tec_lis = atlas.search_relevant_technique(query=test_query, top_k=5, filter_parent="both")
    print("検索結果", [tec.id for tec in searched_tec_lis])


if __name__ == "__main__":
    main()
