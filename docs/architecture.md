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
  "target_source": "osu-wiki",
  "retrieval_lane": "canonical"
}
```

`canonical_document_id`, `topic_document_ids`, `target_source`, and `retrieval_lane` are optional. This lets GLiNER or a later classifier enrich candidate aliases offline while deterministic validation, confidence thresholds, and ambiguity checks protect the serving path.

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

## Feedback loop

Discord reactions can be logged as evaluation data: query, analysis, retrieved chunk IDs, answer version, and explicit positive or negative feedback. Do not mutate routing weights live from a single reaction. Review and aggregate feedback offline, add representative evaluation cases, then promote a new alias artifact or analyzer only when the benchmark and regression tests improve.
