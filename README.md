# NER-OpenVINO-APP

## 概要

このリポジトリは、日本語の文章から人名や企業名などの「固有表現」を見つけ出し、必要に応じて抽象化や匿名化を行うデモアプリケーションです。主な目的は、誰でも簡単に固有表現認識（Named Entity Recognition, NER）を試せるようにすることです。内部では、Intel 社の推論エンジン **OpenVINO** を利用してモデルを高速化し、ゆくゆくはテキスト処理フレームワーク **LangChain** を通じて匿名化・要約を実現する予定です。このアプリは研究用の試作段階であり、本番環境での利用には注意してください。

## 技術スタック

- **プログラミング言語 / 実行環境**
  - Python 3.10 で実装されています。
  - [uv](https://docs.astral.sh/uv/) を用いてパッケージのインストールと仮想環境を管理します。

- **ライブラリ / フレームワーク**
  - [OpenVINO](https://docs.openvino.ai/) : Intel が提供する推論用ライブラリで、モデルの高速化に利用します。
  - [Optimum](https://huggingface.co/docs/optimum/index) : Hugging Face が提供する拡張ライブラリで、OpenVINO 向けのモデル最適化を簡単に行えます。
  - [Transformers](https://huggingface.co/docs/transformers) : 多言語対応の深層学習モデルを扱うためのライブラリです。
  - [pytest](https://docs.pytest.org/) : ユニットテストの実行に使用します。
  - その他、ログ出力や評価用に一部の補助ライブラリを利用します。

## クイックスタート

以下の手順は、Ubuntu や macOS 環境を想定しています。Windows の場合はコマンドが若干異なる場合があります。

1. **リポジトリを取得**

   ```bash
   git clone https://github.com/Stone5656/NER-can-use-NPU-Test.git
   cd NER-can-use-NPU-Test


2. **仮想環境の作成**

   `uv` を利用して Python3.10 の仮想環境を作成します。`uv` がインストールされていない場合は `pip install uv` で導入してください。

3. **依存パッケージのインストール**

   OpenVINO などのライブラリをまとめてインストールします。

   ```bash
   uv sync
   ```

4. **サンプルコードの実行**

   現在は基本的なセットアップのみですが、下記のコマンドでユニットテストを実行し、環境が整っているか確認できます。

   ```bash
   uv run pytest -q
   ```

5. **モデル変換と推論**

   推奨モデルをダウンロードし、OpenVINO 形式に変換したうえで推論を実行できます。
   ```bash
   uv run uvicorn ner_openvino.main:app
   ```

## プロジェクト構成

```
.
├── README.md              # このファイル
├── main.py                # アプリケーションのエントリポイント（今後拡張予定）
├── import_test.py         # インストール済みライブラリの確認用スクリプト
├── logs/                  # ログの出力先（アプリ実行時に生成）
├── tests/                 # pytest 用テストコード
│   └── test_logger_utils.py
├── utils/                 # 共通ユーティリティ
│   └── logger_utils/      # ログ機能のインタフェースと実装
└── pyproject.toml         # uv 用の設定ファイル
```

それぞれのファイルやディレクトリは、モデルの最適化や推論パイプラインを構築するための土台となっています。

## 推奨モデル

日本語の固有表現認識タスクには、Hugging Face で公開されている **tsmatz/xlm-roberta-ner-japanese** モデルの利用を推奨します。このモデルは多言語対応モデル **XLM-RoBERTa** を日本語の固有表現認識向けに微調整したもので、Wikipedia 日本語版の記事に基づくデータセットを用いて訓練されています。モデルは人物 (PER)、組織 (ORG)、場所 (LOC) など複数のカテゴリを予測できます。

推論時には下記のように `transformers` ライブラリの `pipeline` を利用すると簡単に試せます。

```python
from transformers import pipeline

model_name = "tsmatz/xlm-roberta-ner-japanese"
classifier = pipeline("token-classification", model=model_name)
result = classifier("鈴井は4月の陽気の良い日に、鈴をつけて北海道のトムラウシへと登った")
print(result)
```

このモデルは MIT ライセンスの下で公開されており、自由に利用・再配布が可能です。なお、固有表現認識モデルの性能はデータや利用環境に依存しますので、精度検証は各自で行ってください。

## 環境変数と設定

このプロジェクトでは、いくつかの振る舞いを環境変数で調整できるようにしています。環境変数を使うことで、設定値をコードに直接書かずに外部ファイルから読み込み、環境ごとに容易に切り替えられるようになります。例えば、**python-dotenv** ライブラリの `load_dotenv()` 関数を使うと、`.env` ファイルに書かれた設定値を一括で読み込み、`os.getenv()` で参照できます。下記はサンプルです。

```
# .env ファイル例
NER_PATTERN_DIR=./ner_pattern
NER_SAVE_DIR=./models/tsmatz_intel
NER_SAVE_DIR_NPU=./models/tsmatz_intel_npu
NER_MAX_SEQ_LEN=256
NER_BATCH_SIZE=4
LOGPATH=./logs/app.log
LOG_LEVEL=INFO
```

### ner_pattern フォルダ

* `ner_allow.txt`: 許可する固有表現のパターン一覧
* `ner_ignore.txt`: 除外する固有表現のパターン一覧

環境変数 `NER_PATTERN_DIR` を設定することで、これらのファイルを任意のディレクトリから読み込めます。

### ロガー設定 (`with_logger` デコレータ)

`with_logger` デコレータを使うと、関数に `logger` 引数が自動的に注入されます。これにより、関数ごとにログ出力を簡単に行えます。

```python
@with_logger("NER-OpenVINO-APP", log_file="LOG_FILE_PATH", env_var="LOG_LEVEL")
def compute_softmax_batch(predictions, *, logger):
    logger.info("softmax 計算を開始しました")
```

* `LOGPATH`: ログの出力先ファイルを指定できます。未設定の場合は `logs/app.log` に保存されます。
* `LOG_LEVEL`: ログのレベルを指定します（DEBUG, INFO, WARNING など）。

`with_logger` を使うと、関数に `logger` 引数を定義していなくても、モジュール全体にロガーを注入する仕組みが動作します。

## ライセンス

このプロジェクトは MIT ライセンスで公開される予定です。MIT ライセンスは、ソフトウェアの利用や改変、再配布をほぼ無制限に許可する寛容なライセンスです。ライセンスファイルには、著作権表示とライセンス文を含める必要があります。リポジトリのルートに `LICENSE` ファイルを用意し、プロジェクトの年と著作権者名を記載してください。ライセンス選択に迷った場合や法的な不安がある場合は、GitHub Docs が「ベスト プラクティスとして、プロジェクトにはライセンス ファイルを含めることをお勧めします」と述べています。また、GitHub 自身はライセンスに関する法的助言は提供しないため、必要に応じて専門家に相談することが推奨されています。

---

この README は専門用語をなるべく避け、概要・技術スタック・クイックスタート・プロジェクト構成・推奨モデル・環境変数と設定・ライセンスといった基本情報を整理しています。不明点や追加したい情報があれば、Issue を通じてご連絡ください。
