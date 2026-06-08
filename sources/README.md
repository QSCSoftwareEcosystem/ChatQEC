# Sources directory

This directory holds the source-of-truth registry for the ChatQEC knowledge base.

## Files

- `registry.jsonl` — live registry, one JSON object per line. Mutated by the
  ingestion pipeline; reviewed via PR.
- `watchlist.yaml` — list of arXiv categories/authors, GitHub repos, YouTube
  channels, and RSS feeds the weekly refresh job pulls from. *(Added in M4.)*
- `sample/registry.sample.jsonl` — illustrative records showing the schema; not
  loaded by `Registry` (you can `cat` it to see what records look like).

## Lifecycle (spec §6)

| Status | Meaning | Returned by retrieval? |
|---|---|---|
| `active` | Current, trusted | Yes |
| `superseded` | A newer version exists | No (unless explicit override) |
| `retracted` | Paper retracted / blog deleted / repo archived | No |
| `stale` | Hasn't been re-checked in N days | Yes, flagged "may be outdated" |
| `pending_review` | Auto-ingested, awaiting human PR approval | No |

Status transitions are validated by `chatqec.ingestion.registry.Registry`;
illegal transitions raise `InvalidTransitionError`.

## CLI

```bash
chatqec sources list                  # table view of all records
chatqec sources show <id>             # detailed view of one record
chatqec sources mark-retracted <id>   # manual lifecycle override
chatqec sources compact --dry-run     # show records eligible for compaction
```
