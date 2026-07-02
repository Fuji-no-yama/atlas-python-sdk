# 概要
MITRE ATLAS についてPythonから気軽にデータを利用できるようにすることを目的としたライブラリです。

# インストール方法
## pipの場合
- インストール方法: `pip install git+https://github.com/Fuji-no-yama/atlas-python-sdk@2.0.0`
- アップグレード方法: `pip install -U git+https://github.com/Fuji-no-yama/atlas-python-sdk@2.0.0`
- 削除方法: `pip uninstall atlas`

## uvの場合
- インストール・アップグレード方法: `uv add git+https://github.com/Fuji-no-yama/atlas-python-sdk --tag 2.0.0`
- 削除方法: `uv remove atlas`

# データフォーマットについて
本ライブラリは公式が新しく採用した v6 形式 (`format-version: 6.0.0`) のデータを唯一のデータソースとして使用します。バージョン指定にはリリース識別子 (例: `"2026.06"`) を **推奨** し、旧 `format-version` (例: `"5.6.0"`) を指定した場合は `atlas/data/manifest.yaml` を経由して同じリリースに含まれる v6 ファイルへ自動的にフォールバックされます。

| 指定文字列の例 | 意味 |
| --- | --- |
| `"2026.06"` | v6 リリース識別子 (推奨) |
| `"5.6.0"`   | 旧 format-version (`manifest.yaml` 経由でリリース `2026.04` に解決) |

利用可能な識別子は `Atlas().get_available_versions()`(release一覧) と `Atlas().get_available_legacy_versions()`(旧format-version一覧) で取得できます。

# 使い方
## インスタンスの作成方法
基本的なインスタンスの作成方法は以下のとおりです。(version は指定しない場合は `manifest.yaml` 先頭の最新リリースになります)
```python
from atlas import Atlas
atlas = Atlas(version="2026.06")   # v6 リリース指定 (推奨)
# atlas = Atlas(version="5.6.0")   # 旧指定でも可 (2026.04 に自動フォールバック)
# atlas = Atlas()                  # 省略時は最新リリース
```
初回実行時には自動でローカルにベクトルDBを作成します(`OPENAI_API_KEY` が環境変数に設定されている場合のみ)。embedding モデルを変えたい・もう一度ベクトルDBを初期化したい場合には以下のように指定します。
(embedding モデルは text-embedding の small と large を選択できます。)
```python
from atlas import Atlas
atlas = Atlas(version="2026.06", emb_model="text-embedding-3-small", initialize_vector=True)
```
ベクトルDBおよび `technique_vector.avro` / `casestudy_vector.avro` はユーザデータディレクトリ配下 (`user_data_dir/atlas/releases/<release>/`) に配置されます。

## 機能一覧

### Atlas オブジェクト
- `version` (str) : ロード済みのリリース (`"v2026.06"` のように `v` プレフィックス付き)
- `release` (str) : ロード済みのリリース識別子 (例 `"2026.06"`)
- `tactic_list` / `technique_list` / `mitigation_list` / `casestudy_list` : 各エンティティのリスト
- `relationships` (list[AtlasRelationship]) : v6形式の relationships を第一級で保持したリスト。`technique.tactics` / `mitigation.technique_list` / `casestudy.procedure` などはここから派生した materialize 済みビュー

### ATLAS リレーションシップ オブジェクト
v6 では関係(edge)がトップレベル `relationships:` に集約されており、本ライブラリでも `atlas.relationships` として第一級で保持します。既存の派生ビュー(`technique.tactics` 等)はロード時に一度 materialize されるため、通常用途ではそちらを使用してください。relationships への直接アクセスは、正規順序 (`sequences`) や関係固有の説明・step-id への到達などに使えます。

#### 保有情報
- `source_id` (str) : 起点エンティティID (例: `AML.CS0000`)
- `target_id` (str) : 対象エンティティID (例: `AML.T0000.001`)
- `type` (Literal) : 関係種別 (`achieves` / `specializes` / `mitigates` / `employs` / `sequences`)
- `description` (str | None) : `mitigates` / `employs` の説明
- `tactic_id` (str | None) : `employs` におけるステップのタクティックID
- `step_id` (str | None) : `employs` における step-id (例 `S00`)
- `leads_to` (list[str] | None) : `employs` における次に遷移し得る step-id 一覧
- `position` (int | None) : `sequences` / `employs` における順序
- `uuid` (str | None) / `references` (list[AtlasReference] | None) : 将来のv6.x で relationship 自体がエンティティ化された場合の受け皿(現行v6.0.0では常にNone)
- `raw` (dict) : 未モデル化フィールド保持用の生辞書

#### 使用例
```python
# タクティックのマトリクス上の正規順序 (kill chain sequence) を取得
matrix_order = [
    r.target_id
    for r in sorted(
        atlas.get_relationships(source_id="ATLAS-matrix", type="sequences"),
        key=lambda r: r.position or 0,
    )
]

# あるテクニックを緩和する mitigation を relationships から逆引き
mitigations_for_tec = atlas.get_relationships(target_id="AML.T0000", type="mitigates")
for rel in mitigations_for_tec:
    print(rel.source_id, "->", rel.target_id, ":", rel.description)
```

`Atlas.get_relationships(source_id=..., target_id=..., type=...)` で絞り込みが可能です。

### ATLAS テクニック オブジェクト

#### 保有情報
- `name` (str) : テクニック名
- `id` (str) : id名 (例 `AML.T0000`)
- `description` (str) : 記述内容(リンクなどを削除し整形済み)
- `have_parent` (bool) : 親がいるかどうか(子テクニックかどうか)
- `parent_id` (str | None) : 親のid名 (例 `AML.T0000`) (親がいない場合は None)
- `description_vector` (list[float] | None) : 記述のベクトル
- `tactics` (list[AtlasTactic]) : 所属するタクティックオブジェクトのリスト
- `uuid` (str | None) : v6 で付与される UUID (v6 データにない場合は None)
- `references` (list[AtlasReference] | None) : 外部参照のリスト
- `platforms` (list[str] | None) : v6 で追加されたプラットフォーム (例: `Predictive AI` / `Generative AI` / `Agentic AI` / `Enterprise`)
- `maturity` (str | None) : 成熟度 (例: `Demonstrated`, `Feasible`)

#### 使用例
atlasテクニックオブジェクトを順番に確認しidが `AML.T0000` のものを取得したい場合
```python
for tec in atlas.technique_list:
    if tec.id == "AML.T0000":
        print("found!!")
        print("platforms:", tec.platforms)
        print("maturity:", tec.maturity)
```

### ATLAS 緩和策 オブジェクト

#### 保有情報
- `name` (str) : 緩和策名
- `id` (str) : id名 (例 `AML.M0000`)
- `description` (str) : 記述内容(リンクなどを削除し整形済み)
- `technique_list` (list[AtlasTechnique]) : 緩和策が紐づいているテクニックのリスト
- `uuid` (str | None) : v6 で付与される UUID
- `references` (list[AtlasReference] | None) : 外部参照のリスト

#### 使用例
atlas緩和策オブジェクトを順番に確認しidが `AML.M0000` について紐づいているテクニックのid一覧を表示したい場合
```python
for mit in atlas.mitigation_list:
    if mit.id == "AML.M0000":
        for tec in mit.technique_list:
            print(tec.id)
```

### ATLAS タクティック オブジェクト

#### 保有情報
- `name` (str) : タクティック名
- `id` (str) : id名 (例 `AML.TA0000`)
- `description` (str) : 記述内容
- `technique_list` (list[AtlasTechnique]) : タクティックに紐づくテクニックのリスト
- `uuid` (str | None) : v6 で付与される UUID
- `references` (list[AtlasReference] | None) : 外部参照のリスト

### ATLAS ケーススタディー オブジェクト

#### 保有情報
- `name` (str) : ケーススタディ名
- `id` (str) : id名 (例 `AML.CS0000`)
- `summary` (str) : 記述概要(v6 では `description` フィールドに相当)
- `procedure` (list[AtlasCaseStudyStep]) : ケーススタディのステップを表すオブジェクトのリスト(`step_id` 昇順にソート済み)
    - `id` (str) : `AML.CS0000.S00` の形式で親ケーススタディid + v6の step-id を結合
    - `step_id` (str | None) : v6 の step-id そのもの (例 `S00`)
    - `technique` (AtlasTechnique) : テクニックオブジェクト
    - `tactic` (AtlasTactic) : タクティックオブジェクト
    - `description` (str) : テクニックの description とは異なりケーススタディ固有の description
    - `leads_to` (list[str] | None) : 次に遷移し得る step-id のリスト
- `target` (str): ケーススタディページに記載されているターゲット情報
- `actor` (str): ケーススタディページに記載されているアクター情報
- `type` (Literal["Exercise", "Incident"]): v6 表記の type
- `reference_title_list` (list[str] | None) : 参考のタイトルリスト (無い場合はNone)
- `reference_url_list` (list[str] | None) : 参考のURLリスト (無い場合はNone)
- `uuid` (str | None) : v6 で付与される UUID
- `references` (list[AtlasReference] | None) : 構造化された外部参照リスト

#### 使用例
atlasケーススタディオブジェクトを順番に確認しidが `AML.CS0000` の summary を確認したい場合
```python
for cs in atlas.casestudy_list:
    if cs.id == "AML.CS0000":
        print(cs.summary)
```

atlasケーススタディオブジェクトを順番に確認しidが `AML.CS0000` の procedure について、tactic の ID、technique の ID、ケースタディ固有の記述について順番に取得する場合
```python
for cs in atlas.casestudy_list:
    if cs.id == "AML.CS0000":
        for step in cs.procedure:
            print("step ID:", step.id)          # 例 "AML.CS0000.S00"
            print("tactic ID:", step.tactic.id)
            print("technique ID:", step.technique.id)
            print("description:", step.description)
            print("leads_to:", step.leads_to)   # 次のstep-id候補 (無ければNone)
```

### テクニック検索

2種類の検索関数を使用することができます。
#### テクニックid検索
idを元にテクニックを検索することができます。(idが存在しない場合はValueErrorをraiseします)
```python
tec = atlas.search_tec_from_id(tec_id="AML.T0000")
print("検索結果", tec.id)
```

#### クエリからのテクニックベクトル検索
記述に類似する内容からベクトル検索を行うことができます。
- query引数: 自由記述のクエリを入力できます。
- top_k引数: 上位何件を取得するかを選択できます。
- filter引数: 子テクニックのみ・親テクニックのみ・両方(全テクニック) の3種類から選択できます。
```python
test_query = "Please search techniques about LLM and RAG"
searched_tec_lis = atlas.search_relevant_technique(query=test_query, top_k=5, filter="both")
print("検索結果", [tec.id for tec in searched_tec_lis])
```

### 緩和策検索

#### 緩和策id検索
idを元に緩和策を検索することができます。(idが存在しない場合はValueErrorをraiseします)
```python
mit = atlas.search_mit_from_id(mit_id="AML.M0000")
print("検索結果", mit.id)
```

### ケーススタディー検索

2種類の検索関数を使用することができます。
#### ケーススタディーid検索
idを元にケーススタディーを検索することができます。(idが存在しない場合はValueErrorをraiseします)
```python
cs = atlas.search_cs_from_id(cs_id="AML.CS0026")
print("検索結果", cs.id)
```

#### クエリからのケーススタディーステップのベクトル検索
クエリを元にケーススタディーのdescriptionに対して最も近い意味を持つステップを返します。
```python
ret = atlas.search_relevant_casestudy(query="RAG LLM API", top_k=3)
for cs_step in ret:
    print("ケーススタディーステップのID", cs_step.id)
    print("ケーススタディーステップが紐づくテクニックのid", cs_step.technique.id)
    print("ケーススタディーステップの親ケーススタディーのid", cs_step.parent_id)
    print("ケーススタディーステップのdescription", cs_step.description)
```

# 破壊的変更(v6対応リファクタリング)
- 公開 import パスが `atlas.core` から **`atlas`** に統一されました。`from atlas import Atlas` を使用してください (`atlas.core` は廃止)。
- データフォーマットが v6 形式 (`format-version: 6.0.0`) の単一 YAML に統一され、`atlas/data/versions/` 以下の旧形式は廃止されました。
- `Atlas(version=...)` の推奨指定がリリース識別子 (例 `"2026.06"`) になりました。旧 `format-version` 文字列 (例 `"5.6.0"`) も引き続き受け付け、`manifest.yaml` 経由で v6 ファイルにフォールバックします。
- `snake_case_name` フィールドは全エンティティから **削除** されました (v6 では YAML アンカーが無いため意味を失いました)。
- エンティティに `uuid` / `references` の Optional フィールドを追加。`AtlasTechnique` にはさらに `platforms` / `maturity` を追加。
- `AtlasCaseStudyStep.id` の形式が `AML.CS0000.0` から **`AML.CS0000.S00`** (v6 の step-id 準拠) に変更。あわせて `step_id` / `leads_to` フィールドを追加。
- `AtlasCaseStudy.type` の Literal 値が `"exercise"/"incident"` から v6 表記の **`"Exercise"/"Incident"`** に変更。
- ベクトル関連ファイル (`technique_vector.avro` / `casestudy_vector.avro` / chromaDB) の格納先が `user_data_dir/atlas/releases/<release>/` に統一。
- v6 の `relationships` を保持する `AtlasRelationship` エンティティと `Atlas.relationships` / `Atlas.get_relationships(...)` を新設 (既存の派生ビュー API と共存)。
