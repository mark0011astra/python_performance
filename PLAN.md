# Python標準 (List) vs NumPy 実務性能検証仕様書（業務網羅版）

## 1. 検証の目的
Python標準実装（List/組み込み関数/math）とNumPy実装（ベクトル化/ufunc/ブロードキャスト）を、実務で頻出する処理単位で比較し、実運用での採用判断に使える定量データを作成する。

特に次の観点を重視する。

1. オーバーヘッド境界: 小規模データでNumPyの固定コストが支配的になる境界を特定する。
2. キャッシュ効率: 行アクセスと列アクセスの差を定量化し、メモリ局所性の影響を明確化する。
3. メモリ確保コスト: In-place更新とコピー更新の差をデータサイズ別に比較する。
4. 前処理パイプライン効果: 標準化、移動平均、クリッピングなど業務前処理の速度差を確認する。
5. 集計/順位処理の実用性: ソート、Top-k、分位点などレポーティング処理の実効速度を評価する。

## 2. 検証環境・前提

* 比較対象:
  * Python: リスト内包表記、組み込み関数（`sum`, `sorted`）、`math`モジュール
  * NumPy: ベクトル化演算、`np.where`, `np.sum`, `np.dot`, `np.convolve`, `np.sort` など
* データ型: 原則 `float64`
* 再現性: 乱数シード固定（`SEED=42`）
* 正当性担保:
  * 各ベンチマークで、計測前にPython実装とNumPy実装の出力一致を検証する
  * 数値は `allclose`（`rtol/atol`）で判定
* 測定方式:
  * `timeit` を使用
  * ウォームアップ後、複数回実行して最小値を採用
  * サイズと処理種別でループ回数を可変化

## 3. データ規模設定

### 3.1 1次元

1. Tiny: `N=100`
2. Medium: `N=10,000`
3. Large: `N=1,000,000`

### 3.2 2次元（`N x N`）

1. Small: `N=100`
2. Medium: `N=500`
3. Large: `N=1,000`

### 3.3 ソート系

1. `N=100`
2. `N=10,000`
3. `N=200,000`

### 3.4 行列積

* `N=100` のみ（Python三重ループの計算量制約により）

## 4. 検証シナリオ一覧

### 4.1 Element-wise（要素演算）

| ID | 検証項目 | Python実装 | NumPy実装 | 業務利用例 |
|:---|:---|:---|:---|:---|
| A1 | スカラー加算 | `[x + 1.5 for x in d]` | `d + 1.5` | 特徴量シフト |
| A2 | スカラー乗算 | `[x * 1.5 for x in d]` | `d * 1.5` | スケーリング |
| A3 | アフィン変換 | `[(x*1.1)+0.3 for x in d]` | `(d*1.1)+0.3` | 線形補正 |
| A4 | クリッピング | `if`分岐で`[-1,1]`へ丸め | `np.clip(d,-1,1)` | 外れ値抑制 |

### 4.2 Math Functions（数学関数）

| ID | 検証項目 | Python実装 | NumPy実装 | 業務利用例 |
|:---|:---|:---|:---|:---|
| B1 | `sin` | `[math.sin(x) for x in d]` | `np.sin(d)` | 信号変換 |
| B2 | `exp` | `[math.exp(x) for x in d]` | `np.exp(d)` | スコア指数化 |
| B3 | `log1p(abs)` | `[math.log1p(abs(x)) for x in d]` | `np.log1p(np.abs(d))` | 対数圧縮 |
| B4 | `sqrt(abs)` | `[math.sqrt(abs(x)+eps) for x in d]` | `np.sqrt(np.abs(d)+eps)` | 強度変換 |

### 4.3 Conditional（条件処理）

| ID | 検証項目 | Python実装 | NumPy実装 | 業務利用例 |
|:---|:---|:---|:---|:---|
| C1 | 2値しきい値 | `[1.0 if x>0 else 0.0 for x in d]` | `np.where(d>0,1,0)` | フラグ生成 |
| C2 | 3値バケット | `if/elif/else` | `np.where`ネスト | リスク区分 |
| C3 | 条件抽出 | `[x for x in d if x>0.5]` | `d[d>0.5]` | 対象抽出 |
| C4 | 負値埋め戻し | `[x if x>0 else 0 for x in d]` | `np.where(d>0,d,0)` | 欠損/負値処理 |

### 4.4 Aggregation（集約）

| ID | 検証項目 | Python実装 | NumPy実装 | 業務利用例 |
|:---|:---|:---|:---|:---|
| D1 | 行和 `axis=1` | `[sum(row) for row in data]` | `np.sum(data,axis=1)` | レコード集計 |
| D2 | 列和 `axis=0` | 列方向総和内包表記 | `np.sum(data,axis=0)` | 特徴量集計 |
| D3 | 行平均 | `[sum(row)/len(row) ...]` | `np.mean(data,axis=1)` | KPI平均 |
| D4 | 行最大 | `[max(row) for row in data]` | `np.max(data,axis=1)` | ピーク検知 |
| D5 | 標準偏差（1D） | 平均→分散→平方根を手計算 | `np.std(d)` | ばらつき特徴量 |
| D6 | 90%分位点（1D） | `sorted` + index | `np.quantile(..., method='nearest')` | KPI閾値 |

### 4.5 Linear Algebra（線形代数）

| ID | 検証項目 | Python実装 | NumPy実装 | 業務利用例 |
|:---|:---|:---|:---|:---|
| E1 | ベクトル内積 | `sum(x*y for x,y in zip(a,b))` | `np.dot(a,b)` | 類似度計算 |
| E2 | 行列×ベクトル | 行ごと内積 | `mat @ vec` | 重み付きスコア |
| E3 | ブロードキャスト加算 | 二重内包+`zip` | `mat + vec` | バイアス加算 |

### 4.6 Preprocessing Pipeline（前処理）

| ID | 検証項目 | Python実装 | NumPy実装 | 業務利用例 |
|:---|:---|:---|:---|:---|
| F1 | Z-score標準化 | 平均/標準偏差計算後に変換 | `(d-mean)/std` | モデル入力標準化 |
| F2 | Min-Max正規化 | `(x-min)/(max-min)` | 同等式のベクトル化 | 入力範囲統一 |
| F3 | 移動平均（窓5） | スライス和の反復 | `np.convolve(..., 'valid')` | 時系列平滑化 |
| F4 | 標準化+クリップ | Z-score後に`[-2,2]`へ丸め | `np.clip((d-mean)/std,-2,2)` | ロバスト前処理 |

### 4.7 Memory/Indexing（メモリ・索引）

| ID | 検証項目 | Python実装 | NumPy実装 | 業務利用例 |
|:---|:---|:---|:---|:---|
| G1 | 連続スライス | `data[start:end]` | `data[start:end]` | バッチ切り出し |
| G2 | ストライドスライス | `data[::2]` | `data[::2]` | ダウンサンプリング |
| G3 | In-place加算 | ループで破壊更新 | `arr += 1` | 更新処理 |
| G4 | Copy加算 | 新リスト再生成 | `arr = arr + 1` | 不変更新 |
| G5 | 連結 | `left + right` | `np.concatenate` | データ結合 |

### 4.8 Sorting/Ranking（並べ替え・順位）

| ID | 検証項目 | Python実装 | NumPy実装 | 業務利用例 |
|:---|:---|:---|:---|:---|
| H1 | 昇順ソート | `sorted(data)` | `np.sort(data)` | レポート整列 |
| H2 | Top-10抽出 | `sorted(..., reverse=True)[:10]` | `np.sort(d)[-10:][::-1]` | 上位指標抽出 |

### 4.9 Matrix Multiplication（行列積）

| ID | 検証項目 | Python実装 | NumPy実装 | 業務利用例 |
|:---|:---|:---|:---|:---|
| I1 | 行列積 | 3重ループ | `A @ B` | 推論コア計算 |

## 5. 出力フォーマット（CSV）

必須カラム:

* `Category`
* `Operation`
* `Size_N`
* `Python_Time_Sec`
* `NumPy_Time_Sec`
* `Speedup_Factor`
* `Note`

追加推奨カラム:

* `Business_Use`

## 6. 分析観点（実務判断用）

1. 小規模境界: A1/A2でTiny時の速度差がどこまで縮小するか
2. キャッシュ影響: D1とD2で`axis=1`/`axis=0`差がどの程度拡大するか
3. メモリ確保コスト: G3とG4でLarge時の差がどこまで広がるか
4. 前処理速度: F1〜F4でパイプライン処理の短縮幅を確認
5. 解析系処理: H1/H2やD6で、レポーティング系タスクの実運用速度を評価
6. 線形代数: E1〜E3、I1でNumPy最適化の効果を確認

## 7. 実行手順

1. 仮想環境を有効化しNumPyを利用可能にする。
2. 次のコマンドを実行する。  
   `./.venv/bin/python benchmark_business_comprehensive.py`
3. 生成物を確認する。
   * `results_comprehensive.csv`
   * `summary_comprehensive.md`

## 8. 合格条件

* 全ID（A1〜I1）が欠損なくCSVに出力される。
* 各IDでPythonとNumPyの出力一致チェックが通過する。
* `summary_comprehensive.md`にカテゴリ別平均と実務チェックポイントが出力される。
