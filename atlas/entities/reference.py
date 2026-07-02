class AtlasReference:
    """
    ATLASオブジェクトが持つ外部参照(タイトル+URL)を表すクラス。

    Args:
        title (str): 参照のタイトル
        url (str): 参照のURL
        ref_id (str | None): 参照識別子(例: "ref-1")、無い場合はNone
    """

    def __init__(self, title: str, url: str, ref_id: str | None = None) -> None:
        self.title: str = title
        self.url: str = url
        self.ref_id: str | None = ref_id
