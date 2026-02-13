#!/usr/bin/env python3
"""
====================================================================
金価格 AI 予測エンジン (Gold Price AI Prediction Engine)
====================================================================
多因子分析 × GradientBoosting による日本円建て金価格の5日間予測

使い方:
  1. 必要なライブラリをインストール:
     pip install yfinance pandas numpy scikit-learn

  2. スクリプトを実行:
     python gold_predictor.py

  3. 出力:
     - predictions.json  → フロントエンド用の予測データ
     - model_report.txt  → モデルの評価レポート
     - コンソールに予測結果を表示

  4. 毎日自動実行する場合（cron / Task Scheduler）:
     crontab -e
     0 10 * * 1-5 cd /path/to/project && python gold_predictor.py

====================================================================
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ====================================================================
# 設定
# ====================================================================
DATA_START = "2018-01-01"
OUTPUT_FILE = "predictions.json"
REPORT_FILE = "model_report.txt"
FORECAST_DAYS = 5  # 何日後を予測するか

# 取得するデータソース
TICKERS = {
    "Gold_USD":  "GC=F",      # 金先物（USD/oz）
    "USDJPY":    "JPY=X",     # 米ドル/円
    "Oil":       "CL=F",      # WTI原油先物
    "SP500":     "^GSPC",     # S&P 500
    "US10Y":     "^TNX",      # 米10年債利回り
    "DXY":       "DX-Y.NYB",  # ドル指数
    "Silver":    "SI=F",      # 銀先物
    "Platinum":  "PL=F",      # 白金先物
    "VIX":       "^VIX",      # 恐怖指数
    "Nikkei":    "^N225",     # 日経225
}


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ====================================================================
# 1. データ取得
# ====================================================================
def fetch_data():
    print_header("📥 データ取得中...")
    
    data = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=DATA_START, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                data[name] = df["Close"]
                print(f"  ✅ {name:12s} ({ticker:10s}): {len(df):>5} 日分")
            else:
                print(f"  ❌ {name:12s} ({ticker:10s}): データなし")
        except Exception as e:
            print(f"  ❌ {name:12s} ({ticker:10s}): エラー - {e}")
    
    return data


# ====================================================================
# 2. 特徴量エンジニアリング
# ====================================================================
def build_features(data):
    print_header("🔧 特徴量を構築中...")
    
    df = pd.DataFrame(data)
    df = df.ffill().dropna(subset=["Gold_USD", "USDJPY"])
    
    # ── 円建て金価格（1グラムあたり）──
    df["Gold_JPY_gram"] = df["Gold_USD"] * df["USDJPY"] / 31.1035
    
    # ── 移動平均線 ──
    for window in [5, 10, 20, 50, 100, 200]:
        df[f"Gold_MA{window}"] = df["Gold_USD"].rolling(window).mean()
    
    # ── 移動平均乖離率 ──
    for window in [20, 50, 100]:
        df[f"Gold_Dev{window}"] = (df["Gold_USD"] - df[f"Gold_MA{window}"]) / df[f"Gold_MA{window}"]
    
    # ── 標準偏差（ボラティリティ） ──
    for window in [10, 20, 50, 100]:
        df[f"Gold_STD{window}"] = df["Gold_USD"].rolling(window).std()
    
    # ── リターン（変化率） ──
    for period in [1, 3, 5, 10, 20, 60]:
        df[f"Gold_Return_{period}d"] = df["Gold_USD"].pct_change(period)
    
    # ── 為替のリターン ──
    for period in [1, 5, 10, 20]:
        df[f"JPY_Return_{period}d"] = df["USDJPY"].pct_change(period)
    
    # ── RSI（相対力指数） ──
    delta = df["Gold_USD"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss))
    
    # ── ボリンジャーバンド位置 ──
    df["BB_Upper"] = df["Gold_MA20"] + 2 * df["Gold_STD20"]
    df["BB_Lower"] = df["Gold_MA20"] - 2 * df["Gold_STD20"]
    df["BB_Position"] = (df["Gold_USD"] - df["Gold_MA20"]) / (2 * df["Gold_STD20"])
    
    # ── MACD ──
    ema12 = df["Gold_USD"].ewm(span=12).mean()
    ema26 = df["Gold_USD"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # ── 年率ボラティリティ ──
    df["Volatility_Annual"] = df["Gold_Return_1d"].rolling(20).std() * np.sqrt(252)
    
    # ── 金銀比率 ──
    if "Silver" in df.columns:
        df["Gold_Silver_Ratio"] = df["Gold_USD"] / df["Silver"]
    
    # ── 金白金比率 ──
    if "Platinum" in df.columns:
        df["Gold_Platinum_Ratio"] = df["Gold_USD"] / df["Platinum"]
    
    # ── ドル指数リターン ──
    if "DXY" in df.columns:
        for period in [1, 5, 20]:
            df[f"DXY_Return_{period}d"] = df["DXY"].pct_change(period)
    
    # ── 原油リターン ──
    if "Oil" in df.columns:
        for period in [1, 5, 20]:
            df[f"Oil_Return_{period}d"] = df["Oil"].pct_change(period)
    
    # ── VIX関連 ──
    if "VIX" in df.columns:
        df["VIX_MA10"] = df["VIX"].rolling(10).mean()
        df["VIX_Change"] = df["VIX"].pct_change(5)
    
    # ── 曜日・月（季節性） ──
    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month
    
    # ── ターゲット：N日後の円建て金価格 ──
    df["Target"] = df["Gold_JPY_gram"].shift(-FORECAST_DAYS)
    
    # 欠損値を除去
    df = df.dropna()
    
    # 特徴量リスト（ターゲットと元データは除外）
    exclude = ["Target", "Gold_JPY_gram", "Gold_USD", "USDJPY",
               "BB_Upper", "BB_Lower"]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    print(f"  📊 データ期間: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  📊 サンプル数: {len(df):,}")
    print(f"  📊 特徴量数:   {len(feature_cols)}")
    
    return df, feature_cols


# ====================================================================
# 3. モデル訓練 & 評価
# ====================================================================
def train_model(df, feature_cols):
    print_header("🤖 モデル訓練中...")
    
    X = df[feature_cols]
    y = df["Target"]
    
    # 時系列分割（80% 訓練 / 20% テスト）
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  訓練データ: {len(X_train):,} サンプル")
    print(f"  テストデータ: {len(X_test):,} サンプル")
    
    # GradientBoosting モデル
    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # テスト予測
    y_pred_test = model.predict(X_test)
    
    # ── 評価指標 ──
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    # 方向精度（上昇/下落を正しく予測した割合）
    actual_prices = df["Gold_JPY_gram"].iloc[split_idx:split_idx + len(y_test)]
    actual_direction = np.sign(y_test.values - actual_prices.values)
    pred_direction = np.sign(y_pred_test - actual_prices.values)
    direction_acc = np.mean(actual_direction == pred_direction) * 100
    
    metrics = {
        "mae": round(float(mae)),
        "rmse": round(float(rmse)),
        "mape": round(float(mape), 2),
        "direction_accuracy": round(float(direction_acc), 1),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(feature_cols),
    }
    
    print(f"\n  📊 テスト評価結果:")
    print(f"     MAE  (平均絶対誤差):  ¥{mae:,.0f}/g")
    print(f"     RMSE (二乗平均誤差):  ¥{rmse:,.0f}/g")
    print(f"     MAPE (平均誤差率):    {mape:.2f}%")
    print(f"     方向精度:              {direction_acc:.1f}%")
    
    # 特徴量重要度
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    
    print(f"\n  🏆 特徴量重要度 TOP 15:")
    for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
        bar = "█" * int(imp * 200)
        print(f"     {i:2d}. {feat:25s} {imp:.4f} {bar}")
    
    return model, X, y, X_test, y_test, y_pred_test, metrics, importance


# ====================================================================
# 4. 予測実行
# ====================================================================
def make_prediction(model, X, df):
    print_header("🔮 予測実行中...")
    
    latest = X.iloc[-1:]
    predicted_jpy = model.predict(latest)[0]
    current_jpy = df["Gold_JPY_gram"].iloc[-1]
    current_usd = df["Gold_USD"].iloc[-1]
    current_usdjpy = df["USDJPY"].iloc[-1]
    change_pct = (predicted_jpy - current_jpy) / current_jpy * 100
    
    prediction = {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "forecast_date": (df.index[-1] + timedelta(days=FORECAST_DAYS + 2)).strftime("%Y-%m-%d"),
        "current_jpy_gram": round(float(current_jpy)),
        "predicted_jpy_gram": round(float(predicted_jpy)),
        "change_pct": round(float(change_pct), 2),
        "current_usd_oz": round(float(current_usd), 2),
        "current_usdjpy": round(float(current_usdjpy), 2),
        "direction": "up" if change_pct > 0 else "down",
    }
    
    is_up = change_pct >= 0
    arrow = "▲" if is_up else "▼"
    
    print(f"  📅 データ日付:    {prediction['date']}")
    print(f"  📅 予測対象日:    {prediction['forecast_date']}")
    print(f"  💰 現在価格:      ¥{current_jpy:,.0f}/g")
    print(f"  💰 USD価格:       ${current_usd:,.2f}/oz")
    print(f"  💱 USD/JPY:       ¥{current_usdjpy:.2f}")
    print(f"  🔮 5日後予測:     ¥{predicted_jpy:,.0f}/g")
    print(f"  {arrow} 変動予測:      {change_pct:+.2f}%")
    
    return prediction


# ====================================================================
# 5. JSON出力（フロントエンド用）
# ====================================================================
def export_json(df, X, model, prediction, metrics, importance, y_test, y_pred_test, X_test):
    print_header("💾 JSONファイルを出力中...")
    
    # ── 価格チャートデータ（直近300日） ──
    recent = df.tail(300).copy()
    recent_pred = model.predict(X.tail(300))
    
    chart_data = []
    for i, (idx, row) in enumerate(recent.iterrows()):
        chart_data.append({
            "date": idx.strftime("%Y-%m-%d"),
            "actual": round(float(row["Gold_JPY_gram"])),
            "predicted": round(float(recent_pred[i])),
            "gold_usd": round(float(row["Gold_USD"]), 2),
            "usdjpy": round(float(row["USDJPY"]), 2),
        })
    
    # ── テスト期間の比較データ ──
    test_data = []
    for i in range(min(len(y_test), 200)):
        test_data.append({
            "date": y_test.index[i].strftime("%Y-%m-%d"),
            "actual": round(float(y_test.iloc[i])),
            "predicted": round(float(y_pred_test[i])),
        })
    
    # ── 特徴量重要度 ──
    # 日本語名マッピング
    name_map = {
        "Oil": "原油価格", "SP500": "S&P 500", "DXY": "ドル指数",
        "US10Y": "米10年債", "Silver": "銀価格", "Platinum": "白金価格",
        "VIX": "恐怖指数(VIX)", "Nikkei": "日経225", "Month": "月",
        "DayOfWeek": "曜日", "RSI_14": "RSI(14日)",
        "MACD": "MACD", "MACD_Signal": "MACDシグナル", "MACD_Hist": "MACDヒストグラム",
        "BB_Position": "ボリンジャー位置",
        "Volatility_Annual": "年率ボラティリティ",
        "Gold_Silver_Ratio": "金銀比率", "Gold_Platinum_Ratio": "金白金比率",
    }
    
    feature_data = []
    for feat, imp in importance.head(20).items():
        # 自動的に日本語名を生成
        ja_name = name_map.get(feat, feat)
        if ja_name == feat:
            ja_name = (feat
                .replace("Gold_MA", "金MA")
                .replace("Gold_STD", "金STD")
                .replace("Gold_Dev", "金乖離率")
                .replace("Gold_Return_", "金リターン")
                .replace("JPY_Return_", "円リターン")
                .replace("DXY_Return_", "DXY変化")
                .replace("Oil_Return_", "原油変化")
                .replace("VIX_MA10", "VIX移動平均")
                .replace("VIX_Change", "VIX変化率")
                .replace("d", "日")
            )
        
        feature_data.append({
            "feature": feat,
            "name_ja": ja_name,
            "importance": round(float(imp) * 100, 2),
        })
    
    # ── 全体をまとめる ──
    output = {
        "generated_at": datetime.now().isoformat(),
        "model": {
            "algorithm": "GradientBoostingRegressor",
            "n_estimators": 500,
            "forecast_days": FORECAST_DAYS,
            "data_start": DATA_START,
        },
        "prediction": prediction,
        "metrics": metrics,
        "features": feature_data,
        "chart_data": chart_data,
        "test_data": test_data[-120:],
    }
    
    output_path = Path(OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ {output_path.absolute()}")
    print(f"     チャートデータ: {len(chart_data)} 日分")
    print(f"     テストデータ:   {len(test_data)} 日分")
    print(f"     特徴量:         {len(feature_data)} 個")
    
    return output


# ====================================================================
# 6. レポート出力
# ====================================================================
def write_report(prediction, metrics, importance):
    report = f"""
================================================================
  金価格 AI 予測レポート
  生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================

【予測結果】
  データ日付:      {prediction['date']}
  予測対象日:      {prediction['forecast_date']}
  現在価格:        ¥{prediction['current_jpy_gram']:,}/g
  5日後予測:       ¥{prediction['predicted_jpy_gram']:,}/g
  変動予測:        {prediction['change_pct']:+.2f}%
  方向:            {'上昇 ▲' if prediction['direction'] == 'up' else '下落 ▼'}
  USD/oz:          ${prediction['current_usd_oz']:,.2f}
  USD/JPY:         ¥{prediction['current_usdjpy']:.2f}

【モデル評価（テスト期間）】
  MAE:             ¥{metrics['mae']:,}/g
  RMSE:            ¥{metrics['rmse']:,}/g
  MAPE:            {metrics['mape']}%
  方向精度:        {metrics['direction_accuracy']}%
  訓練サンプル:    {metrics['train_samples']:,}
  テストサンプル:  {metrics['test_samples']:,}
  特徴量数:        {metrics['n_features']}

【特徴量重要度 TOP 15】
"""
    for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
        bar = "█" * int(imp * 150)
        report += f"  {i:2d}. {feat:30s} {imp*100:.2f}%  {bar}\n"
    
    report += """
================================================================
  ⚠️ 注意: この予測は学習・研究目的です。
  投資判断の唯一の根拠にしないでください。
================================================================
"""
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n  📄 レポート: {Path(REPORT_FILE).absolute()}")


# ====================================================================
# メイン
# ====================================================================
def main():
    print("\n" + "🏆" * 20)
    print("  金価格 AI 予測エンジン v2.0")
    print("  Multi-Factor GradientBoosting Model")
    print("🏆" * 20)
    
    # 1. データ取得
    data = fetch_data()
    
    if "Gold_USD" not in data or "USDJPY" not in data:
        print("\n❌ 必須データ（金価格・為替）が取得できませんでした。")
        print("   インターネット接続を確認してください。")
        return
    
    # 2. 特徴量構築
    df, feature_cols = build_features(data)
    
    # 3. モデル訓練
    model, X, y, X_test, y_test, y_pred_test, metrics, importance = \
        train_model(df, feature_cols)
    
    # 4. 予測
    prediction = make_prediction(model, X, df)
    
    # 5. JSON出力
    export_json(df, X, model, prediction, metrics, importance, y_test, y_pred_test, X_test)
    
    # 6. レポート出力
    write_report(prediction, metrics, importance)
    
    print_header("✅ 完了！")
    print(f"  → {OUTPUT_FILE} をフロントエンドHTMLと同じフォルダに配置してください")
    print(f"  → ブラウザでHTMLファイルを開くと予測結果が表示されます\n")


if __name__ == "__main__":
    main()
