# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffolding (uv, Hydra, Click CLI shell, pytest, ruff, mypy)
- ChatQEC public API stub returning placeholder Answer objects
- GitLab CI with lint + test stages
- Pre-commit hooks
- Design spec and implementation plan
- Source registry (`sources/registry.jsonl`) with typed `SourceRecord`, lifecycle states (`active|superseded|retracted|stale|pending_review`), and validated state transitions
- SHA-256 content-hash utilities (`chatqec.ingestion.hashing`)
- Canonical source-ID generators per source type (`chatqec.ingestion.ids`)
- CLI: `chatqec sources list/show/mark-retracted/compact`
- Canonical wiki layer: `CanonicalFrontmatter` schema, directory loader with cross-reference validation, section-based chunker
- 5 P0 canonical pages: surface-code, stabilizer-formalism, threshold-theorem, css-codes, magic-state-distillation
- 52-page canonical roster documentation (`knowledge/canonical/_roster.yaml`)
- CLI: `chatqec canonical list/validate`
- CI: validate-canonical stage runs on every push
