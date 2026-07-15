# osu! Chatbot Evaluation Sets

Seed retrieval evals live in JSONL files. Each row is compatible with:

```powershell
osu-bot eval eval/osu_seed.jsonl
osu-bot eval eval/osu_seed.jsonl --output artifacts/rag/eval_dense_report.json
```

The seed set intentionally mixes direct osu! terminology with natural user phrasing, vague support requests, and symptom descriptions. Expectations are document-level by default so the set stays useful if chunk boundaries change.

Each row may distinguish a primary expectation from explicitly accepted evidence:

```json
{
  "question": "what files or folders does osu put on my machine?",
  "expected_document_ids": ["Client/Installation"],
  "acceptable_document_ids": ["Client/Program_files"]
}
```

`expected_document_ids` remain the strict benchmark. `acceptable_document_ids` are
curated, exact alternatives that contain enough evidence to answer the question.
The evaluator does not infer parent/child relationships from path prefixes. Reports
therefore expose both relation-aware metrics (`matches`, `retrieval_accuracy`) and
the original exact metrics (`strict_matches`, `strict_retrieval_accuracy`).
