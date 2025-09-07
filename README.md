# NER-OpenVINO-APP

日本語固有表現認識 (NER) モデルを OpenVINO で最適化し、LangChain に組み込んで
「固有表現に基づくテキストの抽象化・匿名化」を行うデモアプリケーション。

---

## 技術スタック

- **言語/環境**
  - Python 3.10
  - [uv](https://docs.astral.sh/uv/) （パッケージ・環境管理）

- **主要ライブラリ**
  - [Optimum](https://huggingface.co/docs/optimum/index)（`optimum[openvino,nncf]`）
    - OpenVINO Runtime での推論最適化
    - NNCF (Neural Network Compression Framework) による量子化
  - [OpenVINO](https://docs.openvino.ai/)
  - [NNCF](https://github.com/openvinotoolkit/nncf)
  - [Transformers](https://huggingface.co/docs/transformers)
  - [huggingface_hub](https://huggingface.co/docs/huggingface_hub)
  - [LangChain](https://python.langchain.com/)（予定）
  - [pytest](https://docs.pytest.org/)（ユニットテスト）
  - [seqeval](https://github.com/chakki-works/seqeval)（NER 評価用、今後追加予定）

---

## プロジェクト構成

```

.
├── README.md
├── import\_test.py          # ライブラリの存在確認＆バージョン確認（logger利用）
├── main.py                 # アプリエントリーポイント（今後拡張）
├── logs/                   # ログファイル出力先
├── tests/                  # pytest 用テストコード
│   └── test\_logger\_utils.py
├── utils/                  # 共通ユーティリティ
│   └── logger\_utils/       # ログ機能（Interface/実装）
└── pyproject.toml          # uv 管理設定

````

---

## 環境構築

```bash
# 仮想環境作成 (Python 3.10)
uv init --name NER-OpenVINO-APP --color always --cache-dir ./.python-cache --app -p 3.10

# 依存ライブラリ導入
uv add "optimum[openvino,nncf]" pytest
````

---

## ログ機能

`utils/logger_utils/` 配下に以下を実装：

* **logger\_interface.py**

  * `LoggerFactory` (Protocol) を定義 → Moc に差し替え可能
* **logger\_utils.py**

  * 実装: 標準出力(色付き) + ファイル出力
  * コンソール出力では `[<色付きLEVEL>]` + `(filename:lineno)` を含む

使用例:

```python
from utils.logger_utils import LoggerFactoryImpl

logger = LoggerFactoryImpl("NER-OpenVINO-APP", log_file="logs/app.log")
logger.info("アプリ起動しました")
```

出力例（コンソール）:

```bash
2025-09-07 20:12:57 [[32mINFO[0m] NER-OpenVINO-APP (main.py:10): アプリ起動しました
```

---

## テスト

pytest を用いてユニットテストを実行できます。

```bash
uv run pytest -q
```

* `tests/test_logger_utils.py`

  * 標準出力に **ANSI カラーコード** が含まれるか
  * `(filename:lineno)` が正しく出力されるか
  * ログファイルに **色なし出力** が残るか
  * Moc ロガー差し替えが可能か

---

## 今後のロードマップ

1. Hugging Face モデル（`tsmatz/xlm-roberta-ner-japanese`）の IR 変換
2. NNCF による 8bit 量子化
3. OpenVINO Runtime で推論・オフセット復元
4. LangChain 経由で匿名化・抽象化デモ
5. 精度/速度比較・評価
