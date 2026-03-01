#!/usr/bin/env python3
"""
マルチAI分析ツール (Multi-AI Analysis Tool)
5つのAIが議題を分析・討論し、各自の提案を提出するツール

対応AI:
  1. ChatGPT (OpenAI)
  2. Claude (Anthropic)
  3. Gemini (Google)
  4. DeepSeek
  5. Grok (xAI)

使い方:
  python multi_ai_analyzer.py --topic "議題" --background "背景情報"
  python multi_ai_analyzer.py --config config.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

OUTPUT_FILE = "analysis_result.json"

# ============================================================
# AI プロファイル定義
# ============================================================
AI_PROFILES = {
    "chatgpt": {
        "id": "chatgpt",
        "name": "ChatGPT",
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "api_base": "https://api.openai.com/v1/chat/completions",
        "color": "#10a37f",
        "icon": "🟢",
        "personality": (
            "あなたは ChatGPT です。実用的で幅広い知識に基づいた分析を行います。"
            "複雑な問題をわかりやすく構造化し、具体的なステップに分解するのが得意です。"
            "バランスの取れた視点を提供し、多角的なアプローチを提案します。"
        ),
    },
    "claude": {
        "id": "claude",
        "name": "Claude",
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_base": "https://api.anthropic.com/v1/messages",
        "color": "#d97706",
        "icon": "🟠",
        "personality": (
            "あなたは Claude です。慎重かつ深い分析を行います。"
            "倫理的な配慮やリスクの洗い出しを重視し、見落とされがちな観点を指摘します。"
            "論理的で誠実な議論を心がけ、不確実な点は正直に認めます。"
        ),
    },
    "gemini": {
        "id": "gemini",
        "name": "Gemini",
        "provider": "google",
        "model": "gemini-2.0-flash",
        "api_key_env": "GOOGLE_API_KEY",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/models/",
        "color": "#4285f4",
        "icon": "🔵",
        "personality": (
            "あなたは Gemini です。データドリブンで研究志向の分析を行います。"
            "最新の研究やトレンドを踏まえた知見を提供し、"
            "技術的な可能性と実現性を科学的な視点から評価します。"
        ),
    },
    "deepseek": {
        "id": "deepseek",
        "name": "DeepSeek",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_base": "https://api.deepseek.com/v1/chat/completions",
        "color": "#5b6ee1",
        "icon": "🟣",
        "personality": (
            "あなたは DeepSeek です。技術的に深い分析とコスト効率の高い解決策を提案します。"
            "オープンソースやエンジニアリングの観点を重視し、"
            "実装の実現可能性と技術的なトレードオフを詳細に評価します。"
        ),
    },
    "grok": {
        "id": "grok",
        "name": "Grok",
        "provider": "xai",
        "model": "grok-2",
        "api_key_env": "XAI_API_KEY",
        "api_base": "https://api.x.ai/v1/chat/completions",
        "color": "#ef4444",
        "icon": "🔴",
        "personality": (
            "あなたは Grok です。大胆で型破りな発想を提案します。"
            "常識にとらわれない視点から問題を捉え直し、"
            "ユニークで革新的なアプローチを提案するのが強みです。"
            "ウィットに富んだ表現も交えつつ、本質を突いた分析を行います。"
        ),
    },
}


# ============================================================
# API呼び出し
# ============================================================
def call_openai_compatible(api_base: str, api_key: str, model: str,
                           system_prompt: str, user_prompt: str,
                           timeout: int = 120) -> str:
    """OpenAI互換API呼び出し (ChatGPT / DeepSeek / Grok)"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.8,
        "max_tokens": 2000,
    }
    resp = requests.post(api_base, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_anthropic(api_key: str, model: str,
                   system_prompt: str, user_prompt: str,
                   timeout: int = 120) -> str:
    """Anthropic Claude API呼び出し"""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.8,
        "max_tokens": 2000,
    }
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers, json=payload, timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def call_google(api_key: str, model: str,
                system_prompt: str, user_prompt: str,
                timeout: int = 120) -> str:
    """Google Gemini API呼び出し"""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 2000,
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_ai(profile: dict, system_prompt: str, user_prompt: str) -> str:
    """各AIのAPIを呼び出す統合関数"""
    api_key = os.environ.get(profile["api_key_env"], "")
    if not api_key:
        raise ValueError(f"{profile['name']}: APIキー ({profile['api_key_env']}) が未設定です")

    provider = profile["provider"]
    model = profile["model"]

    if provider == "anthropic":
        return call_anthropic(api_key, model, system_prompt, user_prompt)
    elif provider == "google":
        return call_google(api_key, model, system_prompt, user_prompt)
    else:
        # OpenAI互換: openai, deepseek, xai
        return call_openai_compatible(
            profile["api_base"], api_key, model, system_prompt, user_prompt,
        )


# ============================================================
# 分析パイプライン
# ============================================================
class MultiAIAnalyzer:
    """5つのAIによる分析・討論・提案パイプライン"""

    def __init__(self, topic: str, background: str, active_ais: Optional[list] = None):
        self.topic = topic
        self.background = background
        self.active_ais = active_ais or list(AI_PROFILES.keys())
        self.results = {
            "topic": topic,
            "background": background,
            "generated_at": datetime.now().isoformat(),
            "phases": {
                "analysis": {},
                "discussion": {},
                "proposals": {},
            },
            "ai_profiles": {},
            "errors": [],
        }
        # 利用可能なAIプロファイルを記録
        for ai_id in self.active_ais:
            p = AI_PROFILES[ai_id]
            self.results["ai_profiles"][ai_id] = {
                "name": p["name"],
                "model": p["model"],
                "color": p["color"],
                "icon": p["icon"],
                "available": bool(os.environ.get(p["api_key_env"], "")),
            }

    def _get_available_ais(self) -> list:
        """APIキーが設定されているAIのリストを返す"""
        available = []
        for ai_id in self.active_ais:
            p = AI_PROFILES[ai_id]
            if os.environ.get(p["api_key_env"], ""):
                available.append(ai_id)
        return available

    def run_phase1_analysis(self):
        """Phase 1: 各AIが独立して分析"""
        print("\n" + "=" * 60)
        print("  📋 Phase 1: 個別分析")
        print("=" * 60)

        available = self._get_available_ais()
        if not available:
            print("  ❌ 利用可能なAIがありません。APIキーを設定してください。")
            return

        for ai_id in available:
            profile = AI_PROFILES[ai_id]
            print(f"\n  {profile['icon']} {profile['name']} 分析中...")

            system_prompt = (
                f"{profile['personality']}\n\n"
                "以下の議題について分析してください。\n"
                "構成:\n"
                "1. 現状認識（問題の本質）\n"
                "2. 重要なポイント（3-5点）\n"
                "3. リスクと機会\n"
                "4. 初期的な方向性\n\n"
                "分析は日本語で、簡潔かつ具体的に行ってください。"
            )
            user_prompt = f"【議題】{self.topic}\n\n【背景】{self.background}"

            try:
                response = call_ai(profile, system_prompt, user_prompt)
                self.results["phases"]["analysis"][ai_id] = {
                    "ai": profile["name"],
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                }
                print(f"  ✅ {profile['name']} 分析完了")
            except Exception as e:
                error_msg = f"{profile['name']}: {str(e)}"
                self.results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")

            time.sleep(1)  # レート制限対策

    def run_phase2_discussion(self):
        """Phase 2: 他のAIの分析を踏まえた討論"""
        print("\n" + "=" * 60)
        print("  💬 Phase 2: 相互討論")
        print("=" * 60)

        analyses = self.results["phases"]["analysis"]
        if len(analyses) < 2:
            print("  ⚠️ 討論には2つ以上のAI分析が必要です")
            return

        # 全分析をまとめる
        all_analyses = ""
        for ai_id, data in analyses.items():
            profile = AI_PROFILES[ai_id]
            all_analyses += f"\n--- {profile['name']} の分析 ---\n{data['content']}\n"

        available = self._get_available_ais()
        for ai_id in available:
            if ai_id not in analyses:
                continue
            profile = AI_PROFILES[ai_id]
            print(f"\n  {profile['icon']} {profile['name']} 討論中...")

            system_prompt = (
                f"{profile['personality']}\n\n"
                "あなたは他のAIたちと討論しています。\n"
                "他のAIの分析を読んで、以下の観点で討論してください：\n"
                "1. 賛同する点とその理由\n"
                "2. 異論・補足がある点とその根拠\n"
                "3. 見落とされている観点\n"
                "4. 他のAIの意見を踏まえた新たな気づき\n\n"
                "建設的な討論を心がけ、日本語で簡潔に議論してください。"
            )
            user_prompt = (
                f"【議題】{self.topic}\n\n"
                f"【背景】{self.background}\n\n"
                f"【他のAIたちの分析】\n{all_analyses}"
            )

            try:
                response = call_ai(profile, system_prompt, user_prompt)
                self.results["phases"]["discussion"][ai_id] = {
                    "ai": profile["name"],
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                }
                print(f"  ✅ {profile['name']} 討論完了")
            except Exception as e:
                error_msg = f"{profile['name']}: {str(e)}"
                self.results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")

            time.sleep(1)

    def run_phase3_proposals(self):
        """Phase 3: 最終提案"""
        print("\n" + "=" * 60)
        print("  🎯 Phase 3: 最終提案")
        print("=" * 60)

        analyses = self.results["phases"]["analysis"]
        discussions = self.results["phases"]["discussion"]

        if not analyses:
            print("  ⚠️ 分析データがありません")
            return

        # 全データをまとめる
        context = ""
        context += "\n=== Phase 1: 各AIの分析 ===\n"
        for ai_id, data in analyses.items():
            profile = AI_PROFILES[ai_id]
            context += f"\n--- {profile['name']} ---\n{data['content']}\n"

        if discussions:
            context += "\n=== Phase 2: 討論 ===\n"
            for ai_id, data in discussions.items():
                profile = AI_PROFILES[ai_id]
                context += f"\n--- {profile['name']} ---\n{data['content']}\n"

        available = self._get_available_ais()
        for ai_id in available:
            if ai_id not in analyses:
                continue
            profile = AI_PROFILES[ai_id]
            print(f"\n  {profile['icon']} {profile['name']} 最終提案作成中...")

            system_prompt = (
                f"{profile['personality']}\n\n"
                "これまでの分析と討論を踏まえて、最終提案を作成してください。\n"
                "以下の構成で提案してください：\n\n"
                "## 提案タイトル\n"
                "（一行で提案の核心を表現）\n\n"
                "## 提案概要\n"
                "（2-3文で要約）\n\n"
                "## 具体的なアクションプラン\n"
                "（3-5つの具体的なステップ）\n\n"
                "## 期待される効果\n\n"
                "## リスクと対策\n\n"
                "## 補足・注意点\n\n"
                "日本語で、実行可能で具体的な提案を心がけてください。"
            )
            user_prompt = (
                f"【議題】{self.topic}\n\n"
                f"【背景】{self.background}\n\n"
                f"【これまでの分析と討論】\n{context}"
            )

            try:
                response = call_ai(profile, system_prompt, user_prompt)
                self.results["phases"]["proposals"][ai_id] = {
                    "ai": profile["name"],
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                }
                print(f"  ✅ {profile['name']} 提案完了")
            except Exception as e:
                error_msg = f"{profile['name']}: {str(e)}"
                self.results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")

            time.sleep(1)

    def run_all(self):
        """全フェーズを実行"""
        print(f"\n{'='*60}")
        print(f"  🤖 マルチAI分析ツール")
        print(f"{'='*60}")
        print(f"  議題: {self.topic}")
        print(f"  背景: {self.background[:80]}...")

        available = self._get_available_ais()
        missing = [
            f"{AI_PROFILES[ai_id]['name']} ({AI_PROFILES[ai_id]['api_key_env']})"
            for ai_id in self.active_ais if ai_id not in available
        ]
        print(f"\n  利用可能AI: {len(available)}/{len(self.active_ais)}")
        for ai_id in available:
            p = AI_PROFILES[ai_id]
            print(f"    {p['icon']} {p['name']} ({p['model']})")
        if missing:
            print(f"\n  ⚠️ 未設定: {', '.join(missing)}")

        self.run_phase1_analysis()
        self.run_phase2_discussion()
        self.run_phase3_proposals()

        self.save_results()
        return self.results

    def save_results(self):
        """結果をJSONファイルに保存"""
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 結果を {OUTPUT_FILE} に保存しました")
        print(f"  🌐 multi_ai_analysis.html をブラウザで開いて結果を表示できます")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="マルチAI分析ツール - 5つのAIが議題を分析・討論・提案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python multi_ai_analyzer.py --topic "新規事業の方向性" --background "当社は..."
  python multi_ai_analyzer.py --topic "技術選定" --background "..." --ais chatgpt claude gemini

環境変数 (APIキー):
  OPENAI_API_KEY     - ChatGPT
  ANTHROPIC_API_KEY  - Claude
  GOOGLE_API_KEY     - Gemini
  DEEPSEEK_API_KEY   - DeepSeek
  XAI_API_KEY        - Grok
        """,
    )
    parser.add_argument("--topic", "-t", required=True, help="分析する議題")
    parser.add_argument("--background", "-b", required=True, help="背景情報・コンテキスト")
    parser.add_argument(
        "--ais", nargs="+", default=list(AI_PROFILES.keys()),
        choices=list(AI_PROFILES.keys()),
        help="使用するAI (デフォルト: 全5つ)",
    )
    parser.add_argument("--output", "-o", default=OUTPUT_FILE, help="出力ファイル名")

    args = parser.parse_args()

    global OUTPUT_FILE
    OUTPUT_FILE = args.output

    analyzer = MultiAIAnalyzer(
        topic=args.topic,
        background=args.background,
        active_ais=args.ais,
    )
    analyzer.run_all()
    print("\n  ✅ 分析完了！\n")


if __name__ == "__main__":
    main()
