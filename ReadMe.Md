# 概要
MITRE ATLAS についてPythonから気軽にデータを利用できるようにすることを目的としたライブラリです。

# インストール方法
- インストール方法: `pip install git+https://github.com/Fuji-no-yama/atlas-python-sdk`
- アップグレード方法: `pip install -U git+https://github.com/Fuji-no-yama/atlas-python-sdk`
- 削除方法: `pip uninstall atlas`

# 使い方
## インスタンスの作成方法
基本的なインスタンスの作成方法は以下のとおりです。(versionは指定しない場合は最新のものになります)
```python
from atlas.core import Atlas
atlas = Atlas(version="4.8.0")
```
初回実行時には自動でローカルにベクトルDBを作成します。ただし、その後にembeddingモデルを変えたい・もう一度ベクトルDBを初期化したいなどの場合には以下の方法で初期化を行えます。  
(embedding modelはtext-embeddingのsmallとlargeを選択できます。)
```python
from atlas.core import Atlas
atlas = Atlas(version="4.8.0", emb_model="text-embedding-3-small", initialize_vector=True)
```
## 機能一覧

<details>
<summary>ATLAS テクニック</summary>

### 保有情報
- name(str) : テクニック名
- id(str) : id名 (例 AML.T0000)
- description(str) : 記述内容(リンクなどを削除し整形済み)
- have_parent(bool) : 親がいるかどうか(子テクニックかどうか)
- parent_id(str) : 親のid名 (例 AML.T0000) (親がいない場合はNone)
- description_vector(list[float]) : 記述のベクトル
- tactics(list[AtlasTactic]) : 所属するtacticオブジェクトのリスト
- snake_case_name(str) : yamlファイルに記載されている名前

### 使用例
atlasテクニックオブジェクトを順番に確認しidがAML.T0000のものを取得したい場合
```python
for tec in atlas.technique_list:
    if tec.id == "AML.T0000":
        print("found!!")
```

</details>

<details>
<summary>ATLAS 緩和策</summary>

### 保有情報
- name(str) : 緩和策名
- id(str) : id名 (例 AML.M0000)
- description(str) : 記述内容(リンクなどを削除し整形済み)
- technique_list(list[AtlasTechnique]) : 緩和策が紐づいているテクニックのリスト
- snake_case_name(str) : yamlファイルに記載されている名前

### 使用例
atlas緩和策オブジェクトを順番に確認しidがAML.M0000について紐づいているテクニックのid一覧を表示したい場合
```python
for mit in atlas.mitigation_list:
    if mit.id == "AML.M0000":
        for tec in mit.technique_list:
        print(tec.id)
```

</details>

<details>
<summary>ATLAS ケーススタディー</summary>

### 保有情報
- name(str) : 緩和策名
- id(str) : id名 (例 AML.CS0000)
- description(str) : 記述内容(リンクなどを削除し整形済み)
- technique_list(list[AtlasTechnique]) : ケーススタディのステップを構成するテクニックのリスト(順番もこのまま)
- target (str): ケーススタディページに記載されているターゲット情報
- actor (str): ケーススタディページに記載されているアクター情報
- type (str): "exercise" または "incident"
- reference_title(str) : 参考のタイトル (記載がない場合はNone)
- reference_url(str) : 参考のURL (記載がない場合はNone)

### 使用例
atlasケーススタディオブジェクトを順番に確認しidがAML.CS0000について紐づいているテクニックのid一覧を順番に表示したい場合
```python
for cs in atlas.casestudy_list:
    if cs.id == "AML.CS0000":
        for tec in cs.technique_list:
        print(tec.id)
```

</details>

<details>
<summary>テクニック検索</summary>

2種類の検索関数を使用することができます。
### テクニックid検索
idを元にテクニックを検索することができます。
```python
tec = atlas.search_tec_from_id(id="AML.T0000")
print("検索結果", tec.id)
```

### クエリからのテクニックベクトル検索
記述に類似する内容からベクトル検索を行うことができます。
- query引数: 自由記述のクエリを入力できます。
- top_k引数: 上位何件を取得するかを選択できます。
- filter引数: 子テクニックのみ・親テクニックのみ・両方(全テクニック) の3種類から選択できます。
```python
test_query = "Please search techniques about LLM and RAG"
searched_tec_lis = atlas.search_relevant_technique(query=test_query, top_k=5, filter_parent="both")
print("検索結果", [tec.id for tec in searched_tec_lis])
```

</details>

<!-- # その他
## ディレクトリ構造

## お問い合わせ -->