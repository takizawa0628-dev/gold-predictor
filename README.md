# 🏆 金価格 AI 予測ダッシュボード

**全自動** の金価格予測システム。毎日 GitHub Actions が自動でデータ取得 → AI予測 → ウェブサイト更新を行います。

## 🌟 特徴

- 🤖 **GradientBoosting** 機械学習モデル（500本の決定木）
- 📊 **10種のデータソース**: 金・為替・原油・株式・債券・VIX 等
- 🧮 **50+の特徴量**: 移動平均・RSI・MACD・ボリンジャーバンド等
- ⏰ **毎日自動更新**: GitHub Actions が平日10時(JST)に実行
- 🌐 **GitHub Pages** で無料ホスティング

## 🚀 セットアップ手順（10分で完了）

### Step 1: このリポジトリをフォーク or 新規作成

GitHub で新しいリポジトリを作成し、以下のファイルをアップロード:

```
📂 your-repo/
├── 📄 index.html              ← フロントエンド
├── 🐍 gold_predictor.py       ← AI予測エンジン
├── 📋 requirements.txt        ← Pythonパッケージ
├── 📖 README.md               ← このファイル
└── 📂 .github/
    └── 📂 workflows/
        └── ⚙️ predict.yml     ← 自動実行設定
```

### Step 2: GitHub Pages を有効化

1. リポジトリの **Settings** → **Pages** に移動
2. **Source** を `Deploy from a branch` に設定
3. **Branch** を `gh-pages` / `/ (root)` に設定
4. **Save** をクリック

### Step 3: GitHub Actions を有効化

1. リポジトリの **Actions** タブに移動
2. "I understand my workflows, go ahead and enable them" をクリック
3. 左側の **Gold Price AI Prediction** を選択
4. **Run workflow** ボタンで初回手動実行

### Step 4: 完了！🎉

- **ウェブサイト**: `https://あなたのユーザー名.github.io/リポジトリ名/`
- 毎日平日10時(JST)に自動更新されます
- 手動更新: Actions タブ → Run workflow

## 📁 ファイル説明

| ファイル | 説明 |
|---------|------|
| `index.html` | ダッシュボードのフロントエンド |
| `gold_predictor.py` | AI予測エンジン（Python） |
| `predictions.json` | 予測結果（自動生成） |
| `model_report.txt` | モデル評価レポート（自動生成） |
| `requirements.txt` | Python依存パッケージ |
| `.github/workflows/predict.yml` | GitHub Actions 設定 |

## ⚠️ 注意事項

- このツールは **学習・研究目的** です
- AIモデルは市場を完璧に予測することはできません
- **投資判断の唯一の根拠にしないでください**
- 投資は自己責任で行ってください

## 🛠️ ローカルで実行する場合

```bash
# インストール
pip install -r requirements.txt

# 予測実行
python gold_predictor.py

# ブラウザで index.html を開く
```

---

Built with ❤️ using Python, scikit-learn, Chart.js & GitHub Actions
