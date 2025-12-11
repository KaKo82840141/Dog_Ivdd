IVDD Gait Classification (DeepLabCut → LSTM)

犬の歩行（DeepLabCut 2DキーポイントCSV）から IVDD (ivdd) / 正常 (normal) の2値分類を行うための学習・評価コードです。
5キーポイント版（従来）に加え、3キーポイント版（left back paw / right back paw / tail set） も同梱しています。

フレーム系列 → スライディングウィンドウ（既定: SEQ_LEN=60, STRIDE=30）

前処理（切替可能）

train1系: z-score（ファイル単位）

train2系: tail_set原点平行移動＋min-max（ファイル単位）

ネットワーク: TimeDistributed(Dense→ReLU) → LSTM → LSTM → Dense(ロジット)

出力は**日付入り（YYYYMMDD-HHMMSS）**で保存

ディレクトリ構成（正）

※ この構成に合わせてコードのパスが固定されています。ivdd/ というフォルダはありません。

.
├── data
│   ├── test
│   │   ├── eval_csv                 # 評価用のDeepLabCut CSV（3段ヘッダ）を入れる
│   │   └── eval_outputs             # 評価の出力先（自動生成）← eval_csv の「兄弟」
│   └── train
│       ├── fig                      # 学習曲線 (loss & accuracy) 画像
│       ├── train_csv                # 学習用のDeepLabCut CSV（3段ヘッダ）を入れる
│       ├── train1_model             # z-score 正規化モデルの保存先
│       ├── train2_model             # tail_minmax 正規化モデルの保存先
│       └── val_misclassified        # 検証で誤分類したウィンドウ一覧CSV
├── environments
│   ├── ML.yaml
│   └── RNNhumanactivityrecognize.yaml
└── scripts
    ├── ivdd_train1.ipynb            # 5KP, z-score 版（既存）
    ├── ivdd_train2.ipynb            # 5KP, tail_minmax 版（既存）
    ├── ivdd_train_3kp.ipynb         # ★ 3KP 学習（z-score / tail_minmax 切替可）
    ├── ivdd_eval.ipynb              # 5KP 評価（既存）
    └── ivdd_eval_3kp.ipynb          # ★ 3KP 評価（z-score / tail_minmax 切替可）

入力データ（DeepLabCut CSV）

3段ヘッダ（例: scorer, bodyparts, coords）

5KP版：left back paw, right back paw, left front paw, right front paw, tail set

3KP版：left back paw, right back paw, tail set

likelihood 列はオプション（ある場合は低品質点を欠損→補間）

キーポイント名の照合は空白/_- を無視して行いますが、基本は上記の英語ラベルで合わせてください。

ラベル推定（ファイル名規約）

normal_...csv → 正常 (0)

ivdd_...csv, ivdd1_...csv, ivdd2_...csv など → IVDD (1)

3KPコードは ivdd1 のような接尾数字も ivdd と判定します。
5KPの古いスクリプトは厳密一致（ivdd / normal）前提の箇所があります。迷ったら 3KP 版の推論関数を参考にしてください。

環境構築
Conda（推奨）
# 例: 環境ファイルから作成
conda env create -f environments/ML.yaml
# or
conda env create -f environments/RNNhumanactivityrecognize.yaml

conda activate <作成した環境名>

主要パッケージ

Python 3.9+（目安）

TensorFlow 2.x（CPU実行に設定済み。GPUを使う場合は後述）

numpy / pandas / scikit-learn / matplotlib

コード先頭で CUDA_VISIBLE_DEVICES=-1 をセットして GPUを無効化しています。GPUを使う場合はその行を削除してください。

使い方（3キーポイント版）
1) 学習（scripts/ivdd_train_3kp.ipynb）

data/train/train_csv/ に学習CSVを配置

ノートブックを開き、冒頭のパス REPO_ROOT をあなたの環境に合わせて修正

正規化方式を選択

NORMALIZE_MODE = "zscore" → train1系（保存先: data/train/train1_model/）

NORMALIZE_MODE = "tail_minmax" → train2系（保存先: data/train/train2_model/）

実行

保存物

学習曲線: data/train/fig/curve_YYYYMMDD-HHMMSS.png

誤分類ウィンドウ: data/train/val_misclassified/val_misclassified_YYYYMMDD-HHMMSS.csv

モデル:

data/train/train1_model/ivdd_lstm_3kp_YYYYMMDD-HHMMSS_best.keras

data/train/train1_model/ivdd_lstm_3kp_YYYYMMDD-HHMMSS_final.keras
（tail_minmax の場合は train2_model/ 配下）

2) 評価（scripts/ivdd_eval_3kp.ipynb）

data/test/eval_csv/ に評価CSVを配置
（eval_outputs を eval_csv の“内側”に置かないでください）

ノートブックを開き、冒頭のパス REPO_ROOT を修正

NORMALIZE_MODE を 学習時と同じ にする

モデル指定

CKPT_PATH に .keras フルパスを指定 または 空文字にしておくと

data/train/train1_model/ → train2_model/ の順で 最新モデルを自動選択

実行

保存物

出力フォルダ: data/test/eval_outputs/YYYYMMDD-HHMMSS/

window_predictions_*.csv

file_level_predictions_*.csv

file_level_errors_*.csv

cm_window_*.png

cm_file_majority_*.png

cm_file_meanprob_*.png

roc_window_*.png

5キーポイント版（既存スクリプト）

scripts/ivdd_train1.ipynb（z-score） → data/train/train1_model/

scripts/ivdd_train2.ipynb（tail_minmax） → data/train/train2_model/

scripts/ivdd_eval.ipynb（評価）

※ 5KPの古いラベル推定は ivdd/normal の厳密一致を仮定している箇所があり、
ivdd1_... のような名前だと「ラベル不明」になる場合があります。
その場合はファイル名を ivdd_... に変更するか、3KP版の推定ロジックを移植してください。

モデル・ハイパーパラメータ（既定）

入力: (T=60, D)（D=10 5KP / D=6 3KP）

事前補正: 欠損補間（線形→前方→後方）、0埋め

正規化:

z-score（ファイル単位, 各次元）

tail_minmax（フレーム毎に tail set を原点へ平行移動 → 各次元を min-max）

ネットワーク:

TimeDistributed(Dense(n_hidden), ReLU)

LSTM(n_hidden, return_sequences=True) → LSTM(n_hidden)

Dense(1)（sigmoidで確率化。2出力の学習物にも評価側で対応済み）

学習:

loss=BinaryCrossentropy(from_logits=True)

optimizer=Adam(lr=1e-4)

ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5)

EarlyStopping は未使用

分割: ファイル単位（リーク防止）

クラス重み: 自動算出（存在クラスが両方ある場合のみ）

よくあるエラーと対処

FileNotFoundError: 評価用CSVが見つかりません
→ data/test/eval_csv/ にCSVが入っているか、EVAL_CSV_DIR のパスを確認。

eval_outputs が eval_csv の内側にできてしまう
→ 評価コードで OUT_BASE = os.path.join(os.path.dirname(EVAL_CSV_DIR), "eval_outputs") を使用。
assert os.path.basename(EVAL_CSV_DIR) == "eval_csv" も入れて誤作成を防止済み。

ValueError: ラベル不明
→ ファイル名に normal または ivdd（ivdd1 など含む）を入れてください。

IndexError: index 1 is out of bounds...（評価時）
→ 学習が Dense(1) のシグモイドなのに、評価で2クラスsoftmax前提の列アクセスをしている場合。
3KP評価コードは logitsの形状（1 or 2）に自動対応しています。

指定キーポイントが見つからない
→ CSVヘッダの bodyparts 名を確認。3KPなら
left back paw / right back paw / tail set を含む必要があります。

GPUを使いたい
→ コード先頭の os.environ["CUDA_VISIBLE_DEVICES"] = "-1" を削除、
かつ GPU対応TFをインストールしてください。

変更ポイント（本リポジトリの実装方針）

すべての成果物（モデル・画像・CSV）に YYYYMMDD-HHMMSS を付与して上書きを回避

評価出力を data/test/eval_outputs/<timestamp>/ に統一（eval_csvの兄弟）

学習曲線は 1枚画像（loss & accuracyの2サブプロット）

誤分類ウィンドウ を学習時検証で data/train/val_misclassified/ に保存

参考：パラメータの主な変更地点

SEQ_LEN, STRIDE: 学習・評価スクリプト先頭付近

キーポイント: KEYPOINTS リスト

正規化方式: NORMALIZE_MODE = "zscore" or "tail_minmax"

ルートパス: REPO_ROOT

ライセンス / 謝辞

DeepLabCut ベースの座標CSVを用いた研究用途のテンプレートです。

本リポジトリのコードは研究・教育用途での使用を想定しています。商用利用時は各依存ライブラリのライセンスをご確認ください。