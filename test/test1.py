from dotenv import load_dotenv

from atlas.core import Atlas


def main() -> None:  # テスト用関数
    load_dotenv(dotenv_path="/workspace/.env.dev", override=True)
    atlas = Atlas(version="5.2.0", emb_model="text-embedding-3-large", initialize_vector=False)

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
    print("ケーススタディーの説明", atlas.casestudy_list[0].summary)
    print("ケーススタディーのステップ", [step.technique.id for step in atlas.casestudy_list[0].procedure])
    print("=================")

    test_query = "Please search techniques about LLM and RAG"
    searched_tec_lis = atlas.search_relevant_technique(query=test_query, top_k=5, filter="both")
    print("検索結果", [tec.id for tec in searched_tec_lis])

    for tec in atlas.technique_list:
        print("=================")
        print(tec.description)


if __name__ == "__main__":
    main()
