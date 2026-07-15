# Architecture and extension contracts

The project keeps corpus understanding, retrieval, and response generation separate. The current implementation uses the osu! wiki and news, but the online path does not depend on either parser.

## Offline source contract

A source adapter should emit document records that the shared chunker can consume. The minimum useful shape is:

```json
{
  "id": "source-stable-document-id",
  "source": "provider-name",
  "source_type": "wiki|news|community|guide|...",
  "title": "Human-readable title",
  "url": "https://canonical-source-location",
  "file_path": "optional/local/source/path",
  "date": "optional ISO date",
  "tags": ["optional", "tags"],
  "metadata": {"source_specific_field": "preserved in chunk payloads"},
  "external_ids": {"provider_id": "123"},
  "relations": [
    {"type": "mentions", "document_id": "Beatmapping/Timing"}
  ],
  "sections": [
    {
      "section_id": "stable-section-id",
      "title": "Optional heading",
      "text": "Normalized plain text"
    }
  ]
}
```

`relations` are optional. An unstructured source may link to a wiki document, another external document, or nothing. Independence is valid; related sources do not need to share a document ID.

The adapter owns scraping and cleanup. The shared chunker owns chunk size, overlap, IDs, and Qdrant payload shape. Very loose text can be represented as one section before more specialized parsing is justified.

## Alias artifact contract

Topic aliases are an offline artifact, not a hand-maintained runtime taxonomy. Each source may contribute JSONL records with these fields:

```json
{
  "alias": "stack leniency",
  "alias_key": "stack leniency",
  "document_id": "Beatmap/Stack_leniency",
  "canonical_document_id": "Beatmap/Stack_leniency",
  "topic_document_ids": ["Beatmap/Stack_leniency"],
  "confidence": 1.0,
  "preference_strength": "strong",
  "target_source": "osu-wiki",
  "retrieval_lane": "canonical"
}
```

`canonical_document_id`, `topic_document_ids`, `preference_strength`, `target_source`, and `retrieval_lane` are optional. `preference_strength` is either `strong` or `soft`: strong topics may reserve several evidence slots, while soft topics reserve only one before general dense retrieval fills the context. When the field is absent, the resolver conservatively infers it from title/path evidence and identifier overlap. This lets GLiNER or a later classifier enrich candidate aliases offline while deterministic validation, confidence thresholds, ambiguity checks, and bounded focus protect the serving path.

The runtime defaults to multi-token, high-confidence aliases. This deliberately avoids hard-routing generic words such as `sound` or `people`. The thresholds are configuration, so a cleaner future corpus may use a different policy.

Run `osu-bot links` followed by `osu-bot aliases` after ingestion to rebuild this artifact from normalized documents and accepted link evidence.

## Online contracts

The online path is:

```text
user query
  -> QueryAnalyzer
  -> QueryAnalysis(intent, resolved topics, retrieval lane)
  -> RetrievalRequest(query, metadata policy)
  -> one dense RetrievalBackend
  -> selected evidence
  -> grounded generator
```

`QueryAnalyzer` is replaceable. Today it combines a small high-precision intent policy with the alias artifact. A trained classifier can replace or wrap it later without changing the dense backend.

`RetrievalRequest` is also backend-neutral. It carries source types, preferred document IDs, and excluded chunk types. Qdrant is the current backend, but these constraints do not expose Qdrant classes to the service layer.

Canonical queries search durable reference sources by default. Temporal queries may also search news. Both lanes are configured as source-type lists.

## Future tools and skills

Tool selection should be a sibling of retrieval, not another retrieval intent label. A future orchestrator can consume `QueryAnalysis` and choose one of three outcomes:

- answer from retrieved evidence;
- call a read-only or state-changing tool under its own authorization policy;
- ask for clarification when required arguments are missing.

The tool router should return a structured action plan. It should not encode tool names into document aliases, and retrieval should remain available as evidence for a tool call when useful.

## Clarification outcome

`QueryAnalysis` may carry a structured `ClarificationRequest` with a reason, prompt, and suggested topic families. The serving path returns this as `response_type: "clarification"`, skips dense retrieval and generation, and exposes the structured prompt through the API.

The current policy is intentionally high precision: it only catches a small set of topic-free requests such as `help me` or `it does not work`. Short but specific questions continue through retrieval. Expand the policy from reviewed feedback, not from broad pronoun or token-count heuristics that could block valid questions.

## Feedback loop

Discord reactions can be logged as source-neutral `FeedbackEvent` records: query, analysis, retrieved chunk and document IDs, answer version, and explicit positive, negative, or corrective feedback. Raw user IDs should not be stored in learning artifacts.

Feedback promotion is a separate offline operation. A `FeedbackReview` must explicitly accept an event and assign canonical topic IDs before it becomes a `QueryTopicExample`; reactions and the analyzer's own predictions never become labels automatically. The promotion command can exclude normalized queries from an evaluation file, and its output omits runtime analysis and retrieved context.

Do not mutate routing weights live from a single reaction. Review and aggregate feedback offline, compare a trained classifier or reranker against the deterministic resolver, then promote a new analyzer only when development metrics and regression tests improve. Once benchmark failures influence training examples, treat that benchmark as a development set and retain a separately collected locked test split for final claims.

The schemas and workflow are documented in `training/README.md`.
