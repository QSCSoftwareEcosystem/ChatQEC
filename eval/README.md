# Eval set

50 QEC Q&A pairs across six subtypes (spec §13).

| Subtype | Count | Example |
|---|---|---|
| definition | 8 | "What is the surface code?" |
| paper_lookup | 8 | "Which paper demonstrated below-threshold surface code performance?" |
| comparison | 9 | "How does the surface code differ from the color code?" |
| math_detail | 9 | "State the Knill-Laflamme correctability conditions." |
| recent_niche | 8 | "What is erasure conversion in neutral-atom QC?" |
| code_execution | 8 | "What does this Stim circuit output: `H 0\nM 0`?" |

## Schema

One JSON object per line in `qa_set.jsonl`:

```json
{
  "id": "def-001",
  "subtype": "definition",
  "question": "...",
  "expected_topics": ["surface-code"],
  "expected_source_id_prefixes": ["canonical:surface-code", "arxiv:2408.13687"],
  "gold_answer_summary": "Short reference answer used by the LLM judge."
}
```

## Running

```bash
chatqec eval run --cases eval/qa_set.jsonl --output eval/results.json
```

The report includes per-subtype recall@K, citation accuracy, and LLM-judged faithfulness.
