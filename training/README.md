# Query-to-topic learning data

`query_topics_seed.jsonl` is the initial reviewed dataset for comparing a semantic topic classifier or reranker with the deterministic alias resolver. Its queries are newly authored paraphrases of observed failure themes; they are not copied from `eval/osu_seed.jsonl`.

Each query-topic row has this shape:

```json
{
  "query": "the circles give me almost no warning before I must click",
  "topic_ids": ["Beatmap/Approach_rate"],
  "negative_topic_ids": ["Gameplay/Hit_object/Approach_circle"],
  "split": "train",
  "source": "reviewed_failure_paraphrase",
  "source_event_id": "seed:001"
}
```

`negative_topic_ids` are optional hard negatives. They are useful for training or evaluating a reranker, but they are not treated as acceptable answers.

## Feedback promotion

Serving surfaces should write append-only feedback events. Do not include raw user IDs or message content other than the query being evaluated.

```json
{
  "event_id": "hashed-or-internal-event-id",
  "occurred_at": "2026-07-15T12:00:00Z",
  "query": "why are my hit windows so strict?",
  "feedback": "negative",
  "source": "discord",
  "analysis": {"topics": []},
  "retrieved_chunk_ids": ["Beatmapping::article"],
  "retrieved_document_ids": ["Beatmapping"],
  "answer_version": "rag-v2"
}
```

A separate offline review file supplies the label. A reaction is never promoted automatically.

```json
{
  "event_id": "hashed-or-internal-event-id",
  "decision": "accept",
  "topic_ids": ["Beatmap/Overall_difficulty"],
  "negative_topic_ids": ["Beatmapping/Timing"],
  "split": "train",
  "notes": "The symptom describes strict judgement windows."
}
```

Promote accepted reviews with:

```powershell
osu-bot build-topic-dataset artifacts/feedback/events.jsonl artifacts/feedback/reviews.jsonl `
  --output artifacts/learning/query_topics_feedback.jsonl `
  --held-out eval/osu_seed.jsonl
```

The held-out guard rejects exact normalized query overlap. Reviewers must still watch for near-duplicates. Because the current evaluation failure themes informed this seed set, `eval/osu_seed.jsonl` is a development benchmark. Create a separately collected, locked `test` split before making production-quality claims.

## Semantic baseline

Compare the deterministic alias resolver with a semantic k-nearest-topic baseline using the existing sentence-transformer configuration:

```powershell
osu-bot eval-topic-model training/query_topics_seed.jsonl `
  --alias-artifact artifacts/rag/alias_build_check/document_aliases.jsonl `
  --output artifacts/learning/topic_model_report.json
```

This is an offline experiment, not a serving-path switch. The semantic model ranks canonical topics by similarity to reviewed training examples and reports coverage, Hit@1, Hit@3, MRR, and hard-negative mistakes on validation or test rows.
