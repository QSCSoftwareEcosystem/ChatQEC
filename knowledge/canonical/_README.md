# Canonical Wiki

This directory holds the hand-curated canonical wiki pages — the highest-authority
content in the ChatQEC corpus. Each page is one QEC concept and gets force-injected
into context whenever a user query matches its `topic_slug` or `aliases`.

## File format

Each page is a markdown file `<topic-slug>.md` with YAML frontmatter:

```yaml
---
topic_slug: surface-code
title: "Surface Code"
aliases: ["toric code variant", "planar code"]
see_also: [stabilizer-formalism, color-code, mwpm-decoder, lattice-surgery]
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers: ["sharmin"]
---
```

### Required fields

- `topic_slug` — kebab-case, must match the filename (`surface-code.md` ↔ `topic_slug: surface-code`)
- `title` — display name
- `authority_tier` — must be `canonical`
- `last_reviewed` — ISO date
- `maintainers` — at least one name

### Optional fields

- `aliases` — alternate names users might query (e.g. "toric code" → surface-code page)
- `see_also` — related slugs; every entry must resolve to a real slug or alias

## Validation

Run `chatqec canonical validate` to check the entire directory. CI runs this on every PR.

## Roster

The full v1 roster of 52 pages (P0/P1/P2 priority) is in `_roster.yaml`. Files starting
with `_` are not loaded as pages — they are documentation / configuration.
