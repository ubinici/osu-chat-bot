# osu! Chatbot Evaluation Sets

Seed retrieval evals live in JSONL files. Each row is compatible with:

```powershell
osu-bot eval eval/osu_seed.jsonl
osu-bot eval eval/osu_seed.jsonl --dense
osu-bot eval eval/osu_seed.jsonl --output artifacts/rag/eval_seed_keyword_report.json
```

The seed set intentionally mixes direct osu! terminology with natural user phrasing, vague support requests, and symptom descriptions. Expectations are document-level by default so the set stays useful if chunk boundaries change.
