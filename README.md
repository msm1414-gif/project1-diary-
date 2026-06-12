# model-router

タスクの内容を渡すと、**Claude Sonnet 4.6 / Opus 4.8 / Fable 5** のどれを使うべきかを
AI が判断してくれる CLI アプリです。

判定そのものは Claude Opus 4.8 が「ルーター（判定器）」として行い、タスクの難易度・
コスト感度・レイテンシ感度・失敗時の影響などを総合して、推奨モデル・次点・信頼度・
理由を構造化出力で返します。

## 判定対象のモデル

| 選択肢 | モデル | 価格 (入力/出力 per 1M) | 想定用途 |
|--------|--------|------------------------|----------|
| `sonnet` | `claude-sonnet-4-6` | $3 / $15 | 速度・コスト重視。要約、分類、抽出、定型コード、大量処理 |
| `opus` | `claude-opus-4-8` | $5 / $25 | 難しい推論、長時間のエージェント処理、複雑なコーディング |
| `fable` | `claude-fable-5` | $10 / $50 | 最難関・高ストークスのタスク。Opus でも厳しい領域 |

## セットアップ

```bash
pip install -e .          # または: pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 使い方

```bash
# 引数でタスクを渡す
model-router "サポートチケットを緊急度で分類する"

# 標準入力から
echo "50万行のモノレポを async に書き換える" | model-router -

# JSON で出力（他ツール連携用）
model-router --json "2ページのメモを3つの箇条書きに要約"
```

出力例:

```
========================================================
  Recommended : sonnet  (claude-sonnet-4-6)
  Runner-up   : opus  (claude-opus-4-8)
  Complexity  : low
  Confidence  : 88%
========================================================

Reasoning:
  要約は明確に定義された定型タスクで、速度とコストを優先すべき。

Considerations:
  - 低い難易度
  - 高いコスト感度
  - レイテンシ重視
```

## ライブラリとして使う

```python
from model_router import recommend_model

rec = recommend_model("複雑な分散システムのバグを原因分析する")
print(rec.recommended_model, rec.model_id, rec.confidence)
```

## 仕組み

- ルーターは Opus 4.8 + adaptive thinking で動作。
- `client.messages.parse()` と Pydantic スキーマ (`ModelRecommendation`) を使い、
  構造化出力で確実にパース可能な結果を得ます。
- 各候補モデルの強み・価格・想定用途をシステムプロンプトに記述し、
  「タスクを快適にこなせる最も安いモデルを既定にし、難易度・ストークス・自律性が
  正当化する場合のみ上位モデルへエスカレートする」方針で判定します。
