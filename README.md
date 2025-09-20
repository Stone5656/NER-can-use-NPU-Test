# NER-OpenVINO-APP 🚀

## 📖 概要

このリポジトリは、日本語の文章から **人名・企業名などの固有表現** を検出し、必要に応じて匿名化や抽象化を行うデモアプリケーションです。
Intel 社の推論エンジン **OpenVINO** を活用し、軽量・高速な処理を目指しています。将来的には **LangChain** と組み合わせて匿名化・要約も可能にする予定です。

⚠️ **研究用の試作段階** です。本番環境で利用する場合は十分にご注意ください。

---

## 🛠 技術スタック

| 分類           | 使用技術                                                 |
| -------------- | -------------------------------------------------------- |
| 言語・実行環境 | Python 3.10                                              |
| 環境管理       | [uv](https://docs.astral.sh/uv/)                         |
| 推論ライブラリ | [OpenVINO](https://docs.openvino.ai/)                    |
| モデル最適化   | [Optimum](https://huggingface.co/docs/optimum/index)     |
| NLP ライブラリ | [Transformers](https://huggingface.co/docs/transformers) |
| テスト         | [pytest](https://docs.pytest.org/)                       |

---

## 🚀 クイックスタート

1. **リポジトリを取得**

   ```bash
   git clone https://github.com/Stone5656/NER-can-use-NPU-Test.git
   cd NER-can-use-NPU-Test
   ```

2. **仮想環境の作成**

   `uv` を利用して Python3.10 の仮想環境を作成します。`uv` がインストールされていない場合は `pip install uv` で導入してください。

3. **依存パッケージのインストール**

   ```bash
   uv sync
   ```

4. **テスト実行**

   ```bash
   uv run pytest -q
   ```

5. **アプリ起動**

   ```bash
   uv run uvicorn ner_openvino.main:app
   ```

   → ブラウザで `http://127.0.0.1:8000/docs` を開くと API ドキュメントが確認できます 🎉

---

## 📂 プロジェクト構成

```
.
├── README.md              # このファイル
├── main.py                # アプリケーションのエントリポイント
├── logs/                  # ログの出力先
├── tests/                 # pytest 用テストコード
├── utils/                 # 共通ユーティリティ
│   └── logger_utils/      # ログ機能
└── pyproject.toml         # uv 設定ファイル
```

---

## 🤖 推奨モデル

- [tsmatz/xlm-roberta-ner-japanese](https://huggingface.co/tsmatz/xlm-roberta-ner-japanese)
  Wikipedia 日本語版で訓練された **XLM-RoBERTa** ベースの NER モデルです。
  人物 (PER)、組織 (ORG)、場所 (LOC) などを検出可能。

```python
from transformers import pipeline
classifier = pipeline("token-classification", model="tsmatz/xlm-roberta-ner-japanese")
print(classifier("京都大学でAIの研究をしています。"))
```

---

## ⚙️ 環境変数と設定

`.env` ファイルに以下を記載することで設定を変更できます。

```dotenv
NER_PATTERN_DIR=./ner_pattern
NER_SAVE_DIR=./models/tsmatz_intel
NER_SAVE_DIR_NPU=./models/tsmatz_intel_npu
NER_MAX_SEQ_LEN=256
NER_BATCH_SIZE=4
LOGPATH=./logs/app.log
LOG_LEVEL=INFO
```

### 📁 ner_pattern フォルダ

- `ner_allow.txt`: 許可する固有表現
- `ner_ignore.txt`: 除外する固有表現
  ➡️ `NER_PATTERN_DIR` を指定すれば任意の場所から読み込み可能。

### 📝 ログ設定 (`with_logger` デコレータ)

関数にロガーを自動注入します。

```python
@with_logger("NER-OpenVINO-APP", log_file="LOG_FILE_PATH", env_log_level="LOG_LEVEL")
def compute_softmax_batch(predictions, *, logger):
    logger.info("softmax 計算開始")
```

- `LOG_FILE_PATH`: 出力先ファイル（未指定なら `logs/app.log`）
- `LOG_LEVEL`: ログレベル（DEBUG, INFO, WARNING…）

---

## 📜 ライセンス

このプロジェクトは **Open Software License** の下で公開されています。

📄 詳細は [LICENSE](./LICENSE) を参照してください。
