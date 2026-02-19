#!/usr/bin/env python3
"""
金価格 AI 予測エンジン v4
- 田中貴金属の店頭小売価格・買取価格をスクレイピング
- 変化率予測（GradientBoosting）
- 円建て予測を田中貴金属ベースで補正
"""

import json
import warnings
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

DATA_START = "2020-01-01"
OUTPUT_FILE = "predictions.json"
REPORT_FILE = "model_report.txt"
FORECAST_DAYS = 1
MAX_CHANGE_PCT = 10.0

TICKERS = {
    "Gold_USD": "GC=F", "USDJPY": "JPY=X", "Oil": "CL=F",
    "SP500": "^GSPC", "US10Y": "^TNX", "DXY": "DX-Y.NYB",
    "Silver": "SI=F", "Platinum": "PL=F", "VIX": "^VIX", "Nikkei": "^N225",
}


def fetch_tanaka_price():
    """田中貴金属の最新金価格をスクレイピング"""
    try:
        import requests
        from bs4 import BeautifulSoup

        url = "https://gold.tanaka.co.jp/commodity/souba/d-gold.php"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        # 最新価格テーブルを探す
        retail_price = None
        buy_price = None
        price_date = None

        # ヘッダーから日付を取得
        h3_tags = soup.find_all("h3")
        for h3 in h3_tags:
            text = h3.get_text()
            if "地金価格" in text:
                match = re.search(r"(\d{4})年(\d{2})月(\d{2})日", text)
                if match:
                    price_date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                break

        # テーブルから価格を取得
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                cell_texts = [c.get_text(strip=True) for c in cells]
                # 「金」の行を探す
                for i, ct in enumerate(cell_texts):
                    if ct == "金" and len(cell_texts) >= i + 3:
                        # 小売価格を探す
                        for j in range(i + 1, len(cell_texts)):
                            price_text = cell_texts[j].replace(",", "").replace("円", "").strip()
                            # 括弧内の前日比を除去
                            price_text = re.sub(r'\(.*?\)', '', price_text).strip()
                            price_text = re.sub(r'[^0-9]', '', price_text)
                            if price_text and len(price_text) >= 4:
                                if retail_price is None:
                                    retail_price = int(price_text[:5] if len(price_text) > 5 else price_text)
                                elif buy_price is None:
                                    buy_price = int(price_text[:5] if len(price_text) > 5 else price_text)
                                    break

        if retail_price and retail_price > 10000:
            print(f"  ✅ 田中貴金属: 小売 ¥{retail_price:,}/g, 買取 ¥{buy_price:,}/g ({price_date})")
            return {
                "retail_price": retail_price,
                "buy_price": buy_price,
                "date": price_date,
                "source": "田中貴金属工業"
            }
        else:
            print(f"  ⚠️ 田中貴金属: 価格取得失敗（パース結果: {retail_price}）")
            return None

    except Exception as e:
        print(f"  ⚠️ 田中貴金属スクレイピング失敗: {e}")
        return None


def fetch_data():
    print("\n" + "=" * 60)
    print("  📥 データ取得中...")
    print("=" * 60)

    # 田中貴金属の価格を取得
    tanaka = fetch_tanaka_price()

    data = {}
    for name, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=DATA_START, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                data[name] = df["Close"]
                print(f"  ✅ {name:12s}: {len(df):>5} 日分")
        except Exception as e:
            print(f"  ❌ {name:12s}: {e}")

    return data, tanaka


def build_features(data):
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

    future_price = df["Gold_JPY_gram"].shift(-FORECAST_DAYS)
    df["Target"] = (future_price - df["Gold_JPY_gram"]) / df["Gold_JPY_gram"] * 100
    df = df.dropna()

    exclude = {"Target", "Gold_JPY_gram", "Gold_USD", "USDJPY",
               "Oil", "SP500", "DXY", "VIX", "Nikkei",
               "Silver", "Platinum", "US10Y"}
    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"\n  📊 サンプル数: {len(df):,} | 特徴量数: {len(feature_cols)}")
    return df, feature_cols


def train_model(df, feature_cols):
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
        "mape": round(float(mape), 2),
        "direction_accuracy": round(float(direction_acc), 1),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(feature_cols),
    }

    print(f"  📊 MAE: ¥{price_mae:,.0f}/g | MAPE: {mape:.2f}% | 方向精度: {direction_acc:.1f}%")

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return model, X, y, X_test, y_test, y_pred_test, metrics, importance, base_prices, pred_prices, actual_prices


def make_prediction(model, X, df, tanaka):
    latest = X.iloc[-1:]
    pred_chg = np.clip(model.predict(latest)[0], -MAX_CHANGE_PCT, MAX_CHANGE_PCT)

    # Yahoo Finance ベースの価格
    yf_current_jpy = df["Gold_JPY_gram"].iloc[-1]
    yf_predicted_jpy = yf_current_jpy * (1 + pred_chg / 100)

    # 田中貴金属ベースの価格（取得できた場合）
    if tanaka and tanaka.get("retail_price"):
        tanaka_retail = tanaka["retail_price"]
        tanaka_buy = tanaka["buy_price"]
        # 田中貴金属の買取価格をベースに予測
        tanaka_predicted_buy = round(tanaka_buy * (1 + pred_chg / 100))
        tanaka_predicted_retail = round(tanaka_retail * (1 + pred_chg / 100))
    else:
        tanaka_retail = None
        tanaka_buy = None
        tanaka_predicted_buy = None
        tanaka_predicted_retail = None

    prediction = {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "forecast_date": (df.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d"),
        "current_jpy_gram": round(float(yf_current_jpy)),
        "predicted_jpy_gram": round(float(yf_predicted_jpy)),
        "change_pct": round(float(pred_chg), 2),
        "current_usd_oz": round(float(df["Gold_USD"].iloc[-1]), 2),
        "current_usdjpy": round(float(df["USDJPY"].iloc[-1]), 2),
        "direction": "up" if pred_chg > 0 else "down",
        # 田中貴金属ベース
        "tanaka_retail_current": tanaka_retail,
        "tanaka_buy_current": tanaka_buy,
        "tanaka_retail_predicted": tanaka_predicted_retail,
        "tanaka_buy_predicted": tanaka_predicted_buy,
        "tanaka_date": tanaka["date"] if tanaka else None,
        "tanaka_source": "田中貴金属工業 店頭価格（税込）",
    }

    arrow = "▲" if pred_chg >= 0 else "▼"
    print(f"\n  🔮 予測結果:")
    if tanaka_buy:
        print(f"     田中貴金属 買取: ¥{tanaka_buy:,}/g → 予測: ¥{tanaka_predicted_buy:,}/g ({arrow}{abs(pred_chg):.2f}%)")
        print(f"     田中貴金属 小売: ¥{tanaka_retail:,}/g → 予測: ¥{tanaka_predicted_retail:,}/g")
    print(f"     国際価格換算:   ¥{yf_current_jpy:,.0f}/g → 予測: ¥{yf_predicted_jpy:,.0f}/g")

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
            "version": "v4 (田中貴金属対応)",
            "n_estimators": 500,
            "forecast_days": FORECAST_DAYS,
        },
        "prediction": prediction,
        "metrics": metrics,
        "features": feature_data,
        "chart_data": chart_data,
        "test_data": test_data[-120:],
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  💾 {OUTPUT_FILE} 保存完了")


def write_report(prediction, metrics, importance):
    report = f"""Gold Price AI Prediction Report v4 (Tanaka Kikinzoku)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Tanaka Buy: Y{prediction.get('tanaka_buy_current', 'N/A'):,}/g -> Y{prediction.get('tanaka_buy_predicted', 'N/A'):,}/g ({prediction['change_pct']:+.2f}%)
International: Y{prediction['current_jpy_gram']:,}/g -> Y{prediction['predicted_jpy_gram']:,}/g
MAE: Y{metrics['mae']:,}/g | MAPE: {metrics['mape']}% | Direction: {metrics['direction_accuracy']}%

Top Features:
"""
    for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
        report += f"  {i:2d}. {feat:30s} {imp*100:.2f}%\n"
    report += "\nDisclaimer: For educational purposes only.\n"

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    print("\n🏆 金価格 AI 予測エンジン v4 (田中貴金属対応)\n")

    data, tanaka = fetch_data()

    if "Gold_USD" not in data or "USDJPY" not in data:
        print("❌ 必須データ取得失敗")
        return

    df, feature_cols = build_features(data)
    result = train_model(df, feature_cols)
    model, X, y, X_test, y_test, y_pred_test, metrics, importance, bp, pp, ap = result

    prediction = make_prediction(model, X, df, tanaka)
    export_json(df, X, model, prediction, metrics, importance, y_test, y_pred_test, bp, pp, ap)
    write_report(prediction, metrics, importance)

    print(f"\n  ✅ 完了！\n")


if __name__ == "__main__":
    main()
