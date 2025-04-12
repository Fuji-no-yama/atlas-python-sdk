"""
ATLASを表現するクラス
現在ケーススタディーについては使用予定がないため作成していない
もし使用する場合は別途yamlファイルからcasestudyの構成を取得する必要あり(関数についてはwork1のdatabase.pyを参照)
"""

import os
import re
from collections.abc import Iterator, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypeVar

import chromadb
import polars as pl
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
        have_parent: bool = False,
        parent_id: str | None = None,
        vector: list[float] | None = None,
        tactics: list[str] | None = None,
    ) -> None:
        self.name = name
        self.id = technique_id
        self.description = self.clean_description(description)  # リンク系統を清掃する
        self.have_parent = have_parent
        self.parent_id = parent_id
        self.description_vector = vector  # ベクトル化されたdesc
        self.tactics = tactics  # list[string]

    def clean_description(self, desc: str) -> str:  # リンクなどのノイズのみを削除する関数
        pattern = r"\[(.*?)\]\(.*?\)"  # リンク表現(表示される部分のみをキャプチャ)
        replacement = r"\1"  # \1で最初のキャプチャグループ(表示される部分)のみに置き換え
        result = re.sub(pattern, replacement, desc)
        return result


class AtlasMitigation:  # 1つの緩和策を表すクラス
    def __init__(self, mitigation_id: str, description: str, tec_lis: list[AtlasTechnique]) -> None:
        self.id = mitigation_id
        self.description = self.clean_description(description)  # リンク系統を清掃する
        self.technique_list = tec_lis

    def show(self) -> None:
        tmp_lis = [tec.id for tec in self.technique_list]
        print(tmp_lis)

    def check_technique_by_id(self, technique_id: str) -> bool:  # ある緩和策内にテクニックが含まれるかを確かめる関数
        return any(tec.id == technique_id for tec in self.technique_list)

    def clean_description(self, desc: str) -> str:
        # リンクなどのノイズのみを削除する関数(mitigationには現在はリンクなどはないが今後に備えてテクニックと同等の関数を用意する)  # noqa: ERA001
        pattern = r"\[(.*?)\]\(.*?\)"  # リンク表現(表示される部分のみをキャプチャ)
        replacement = r"\1"  # \1で最初のキャプチャグループ(表示される部分)のみに置き換え
        result = re.sub(pattern, replacement, desc)
        return result


class Atlas:  # Atlasの機能を保持したクラス
    def __init__(
        self,
        *,  # 以下をキーワード引数に
        version: Literal["4.7.0", "4.8.0"] = "4.8.0",
        emb_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-large",
        initialize_vector: bool = False,
    ) -> None:
        self.version = f"v{version}"
        self.data_dir_path = files("atlas.data").joinpath("versions").joinpath(f"{self.version}")  # パッケージ内のdataディレクトリ
        self.user_data_dir_path = Path(user_data_dir("atlas")) / "vestions" / self.version  # ユーザ側dataディレクトリ
        self.user_data_dir_path.mkdir(parents=True, exist_ok=True)
        self.technique_list = self.__create_tec_list()
        self.mitigation_list = self.__create_mit_list()
        self.chroma_client = chromadb.PersistentClient(str(self.user_data_dir_path.joinpath("chroma")))

        if initialize_vector:
            self.__initialize_vector(model=emb_model)
            self.technique_list = self.__create_tec_list()  # ベクトルを新しい物に置き換えて再実行
            self.mitigation_list = self.__create_mit_list()  # ベクトルを新しい物に置き換えて再実行
        self.chroma_collection = self.__get_chroma_collection(model=emb_model)

    def __create_tec_list(self) -> list[AtlasTechnique]:  # テクニックのリストを作成する関数(Atlasクラスのinitで使用)
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

    def __create_mit_list(self) -> list[AtlasMitigation]:  # Atlas用のmitigationリスト作成関数
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
        try:
            self.chroma_client.delete_collection(name="atlas_technique")  # 存在する場合は一度削除してリセット
            print("コレクションを削除しました。")
        except NotFoundError:
            print("コレクションは存在しません。")
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
    print("テクニックのID:", atlas.technique_list[0].id)
    print("テクニックの名前:", atlas.technique_list[0].name)
    print("テクニックの説明:", atlas.technique_list[0].description)
    test_query = "Please search techniques about LLM and RAG"
    seaerched_tec_lis = atlas.search_relevant_technique(query=test_query, top_k=5, filter_parent="both")
    print("検索結果", [tec.id for tec in seaerched_tec_lis])
    print("一覧", [tec.id for tec in atlas.technique_list])


if __name__ == "__main__":
    main()
