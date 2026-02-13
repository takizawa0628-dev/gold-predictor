#!/usr/bin/env python3
"""
====================================================================
金価格 AI 予測エンジン v3 (Gold Price AI Prediction Engine)
====================================================================
多因子分析 × GradientBoosting による日本円建て金価格の5日間予測

改善点 (v3):
  - 絶対価格ではなく「変化率」を予測（非現実的な予測を防止）
  - 予測変化率に上限/下限を設定（±10%）
  - より安定した特徴量設計

使い方:
  pip install yfinance pandas numpy scikit-learn
  python gold_predictor.py
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
DATA_START = "2020-01-01"
OUTPUT_FILE = "predictions.json"
REPORT_FILE = "model_report.txt"
FORECAST_DAYS = 5

TICKERS = {
    "Gold_USD":  "GC=F",
    "USDJPY":    "JPY=X",
    "Oil":       "CL=F",
    "SP500":     "^GSPC",
    "US10Y":     "^TNX",
    "DXY":       "DX-Y.NYB",
    "Silver":    "SI=F",
    "Platinum":  "PL=F",
    "VIX":       "^VIX",
    "Nikkei":    "^N225",
}

MAX_CHANGE_PCT = 10.0


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def fetch_data():
    print_header("Data fetch")
    data = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=DATA_START, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                data[name] = df["Close"]
                print(f"  OK {name:12s} ({ticker:10s}): {len(df):>5} days")
        except Exception as e:
            print(f"  NG {name:12s} ({ticker:10s}): {e}")
    return data


def build_features(data):
    print_header("Feature engineering")

    df = pd.DataFrame(data)
    df = df.ffill().dropna(subset=["Gold_USD", "USDJPY"])
    df["Gold_JPY_gram"] = df["Gold_USD"] * df["USDJPY"] / 31.1035

    for w in [5, 10, 20, 50, 100]:
        df[f"Gold_MA{w}"] = df["Gold_USD"].rolling(w).mean()

    for w in [5, 20, 50, 100]:
        ma = df["Gold_USD"].rolling(w).mean()
        df[f"Gold_Dev{w}"] = (df["Gold_USD"] - ma) / ma * 100

    for p in [1, 3, 5, 10, 20]:
        df[f"Gold_Ret_{p}d"] = df["Gold_USD"].pct_change(p) * 100

    for p in [1, 5, 10, 20]:
        df[f"JPY_Ret_{p}d"] = df["USDJPY"].pct_change(p) * 100

    delta = df["Gold_USD"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss))

    ma20 = df["Gold_USD"].rolling(20).mean()
    std20 = df["Gold_USD"].rolling(20).std()
    df["BB_Position"] = (df["Gold_USD"] - ma20) / (2 * std20)

    ema12 = df["Gold_USD"].ewm(span=12).mean()
    ema26 = df["Gold_USD"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    df["Volatility"] = df["Gold_USD"].pct_change().rolling(20).std() * np.sqrt(252) * 100

    if "Silver" in df.columns:
        df["Gold_Silver_Ratio"] = df["Gold_USD"] / df["Silver"]
    if "Platinum" in df.columns:
        df["Gold_Platinum_Ratio"] = df["Gold_USD"] / df["Platinum"]

    for asset in ["Oil", "SP500", "DXY", "VIX", "Nikkei"]:
        if asset in df.columns:
            for p in [1, 5, 20]:
                df[f"{asset}_Ret_{p}d"] = df[asset].pct_change(p) * 100

    if "US10Y" in df.columns:
        df["US10Y_Level"] = df["US10Y"]
        df["US10Y_Change_5d"] = df["US10Y"].diff(5)

    df["DayOfWeek"] = df.index.dayofweek
    df["Month"] = df.index.month

    # Target: N-day forward CHANGE RATE (%)
    future_price = df["Gold_JPY_gram"].shift(-FORECAST_DAYS)
    df["Target"] = (future_price - df["Gold_JPY_gram"]) / df["Gold_JPY_gram"] * 100

    df = df.dropna()

    exclude = {"Target", "Gold_JPY_gram", "Gold_USD", "USDJPY",
               "Oil", "SP500", "DXY", "VIX", "Nikkei",
               "Silver", "Platinum", "US10Y"}
    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"  Period: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Samples: {len(df):,}")
    print(f"  Features: {len(feature_cols)}")

    return df, feature_cols


def train_model(df, feature_cols):
    print_header("Model training")

    X = df[feature_cols]
    y = df["Target"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, min_samples_split=15, min_samples_leaf=8,
        max_features=0.6, random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred_test = np.clip(model.predict(X_test), -MAX_CHANGE_PCT, MAX_CHANGE_PCT)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    actual_dir = np.sign(y_test.values)
    pred_dir = np.sign(y_pred_test)
    direction_acc = np.mean(actual_dir == pred_dir) * 100

    base_prices = df["Gold_JPY_gram"].iloc[split_idx:split_idx + len(y_test)].values
    actual_prices = base_prices * (1 + y_test.values / 100)
    pred_prices = base_prices * (1 + y_pred_test / 100)
    price_mae = np.mean(np.abs(actual_prices - pred_prices))
    mape = np.mean(np.abs(actual_prices - pred_prices) / actual_prices) * 100

    metrics = {
        "mae": round(float(price_mae)),
        "rmse": round(float(rmse), 3),
        "mape": round(float(mape), 2),
        "direction_accuracy": round(float(direction_acc), 1),
        "change_rate_mae": round(float(mae), 3),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(feature_cols),
    }

    print(f"  Price MAE: Y{price_mae:,.0f}/g | MAPE: {mape:.2f}% | Direction: {direction_acc:.1f}%")

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return model, X, y, X_test, y_test, y_pred_test, metrics, importance, base_prices, pred_prices, actual_prices


def make_prediction(model, X, df):
    print_header("Prediction")

    latest = X.iloc[-1:]
    pred_chg = np.clip(model.predict(latest)[0], -MAX_CHANGE_PCT, MAX_CHANGE_PCT)

    current_jpy = df["Gold_JPY_gram"].iloc[-1]
    predicted_jpy = current_jpy * (1 + pred_chg / 100)

    prediction = {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "forecast_date": (df.index[-1] + timedelta(days=FORECAST_DAYS + 2)).strftime("%Y-%m-%d"),
        "current_jpy_gram": round(float(current_jpy)),
        "predicted_jpy_gram": round(float(predicted_jpy)),
        "change_pct": round(float(pred_chg), 2),
        "current_usd_oz": round(float(df["Gold_USD"].iloc[-1]), 2),
        "current_usdjpy": round(float(df["USDJPY"].iloc[-1]), 2),
        "direction": "up" if pred_chg > 0 else "down",
    }

    arrow = "▲" if pred_chg >= 0 else "▼"
    print(f"  Current: Y{current_jpy:,.0f}/g -> Predicted: Y{predicted_jpy:,.0f}/g ({arrow}{abs(pred_chg):.2f}%)")

    return prediction


def export_json(df, X, model, prediction, metrics, importance,
                y_test, y_pred_test, base_prices, pred_prices, actual_prices):

    recent = df.tail(300).copy()
    recent_X = X.tail(300)
    recent_pred_chg = np.clip(model.predict(recent_X), -MAX_CHANGE_PCT, MAX_CHANGE_PCT)

    chart_data = []
    gold_jpy = recent["Gold_JPY_gram"].values
    for i, (idx, row) in enumerate(recent.iterrows()):
        actual = gold_jpy[i]
        pred = actual * (1 + recent_pred_chg[i] / 100)
        chart_data.append({
            "date": idx.strftime("%Y-%m-%d"),
            "actual": round(float(actual)),
            "predicted": round(float(pred)),
            "gold_usd": round(float(row["Gold_USD"]), 2),
            "usdjpy": round(float(row["USDJPY"]), 2),
        })

    test_data = []
    for i in range(min(len(y_test), 200)):
        test_data.append({
            "date": y_test.index[i].strftime("%Y-%m-%d"),
            "actual": round(float(actual_prices[i])),
            "predicted": round(float(pred_prices[i])),
        })

    name_map = {
        "RSI_14": "RSI(14日)", "BB_Position": "ボリンジャー位置",
        "MACD": "MACD", "MACD_Signal": "MACDシグナル",
        "MACD_Hist": "MACDヒストグラム", "Volatility": "ボラティリティ",
        "Gold_Silver_Ratio": "金銀比率", "Gold_Platinum_Ratio": "金白金比率",
        "US10Y_Level": "米10年債利回り", "US10Y_Change_5d": "米10年債変化",
        "DayOfWeek": "曜日", "Month": "月",
    }

    feature_data = []
    for feat, imp in importance.head(20).items():
        ja = name_map.get(feat, feat)
        if ja == feat:
            ja = (feat
                .replace("Gold_MA", "金MA").replace("Gold_Dev", "金乖離率")
                .replace("Gold_Ret_", "金リターン").replace("JPY_Ret_", "円リターン")
                .replace("Oil_Ret_", "原油変化").replace("SP500_Ret_", "S&P500変化")
                .replace("DXY_Ret_", "DXY変化").replace("VIX_Ret_", "VIX変化")
                .replace("Nikkei_Ret_", "日経変化").replace("d", "日")
            )
        feature_data.append({
            "feature": feat, "name_ja": ja,
            "importance": round(float(imp) * 100, 2),
        })

    output = {
        "generated_at": datetime.now().isoformat(),
        "model": {
            "algorithm": "GradientBoostingRegressor",
            "version": "v3",
            "n_estimators": 500,
            "forecast_days": FORECAST_DAYS,
            "data_start": DATA_START,
            "max_change_pct": MAX_CHANGE_PCT,
        },
        "prediction": prediction,
        "metrics": metrics,
        "features": feature_data,
        "chart_data": chart_data,
        "test_data": test_data[-120:],
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {OUTPUT_FILE} ({len(chart_data)} chart days, {len(feature_data)} features)")


def write_report(prediction, metrics, importance):
    report = f"""Gold Price AI Prediction Report v3
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Prediction: Y{prediction['current_jpy_gram']:,}/g -> Y{prediction['predicted_jpy_gram']:,}/g ({prediction['change_pct']:+.2f}%)
Price MAE: Y{metrics['mae']:,}/g | MAPE: {metrics['mape']}% | Direction: {metrics['direction_accuracy']}%

Top Features:
"""
    for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
        report += f"  {i:2d}. {feat:30s} {imp*100:.2f}%\n"
    report += "\nDisclaimer: For educational purposes only. Not investment advice.\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    print("\n Gold Price AI Prediction Engine v3\n")
    data = fetch_data()

    if "Gold_USD" not in data or "USDJPY" not in data:
        print("ERROR: Required data not available.")
        return

    df, feature_cols = build_features(data)
    result = train_model(df, feature_cols)
    model, X, y, X_test, y_test, y_pred_test, metrics, importance, bp, pp, ap = result

    prediction = make_prediction(model, X, df)
    export_json(df, X, model, prediction, metrics, importance, y_test, y_pred_test, bp, pp, ap)
    write_report(prediction, metrics, importance)

    print(f"\n  Done! Open index.html to view results.\n")


if __name__ == "__main__":
    main()
