# ChatQEC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ChatQEC v1 — an open-source semi-automated research assistant that answers Quantum Error Correction questions over a heterogeneous corpus, with citations and sandboxed Stim execution, exposed as a Python library, CLI, and Streamlit app.

**Architecture:** Hybrid RAG (Qdrant vector + BM25 + cross-encoder reranker) over a JSONL source registry, with a hand-curated canonical wiki layer that's force-injected on topic match. Claude Sonnet synthesizes with native tool-use, calling a sibling `chatqec-mcp-tools` MCP server (Stim in v1) in a Docker sandbox. Gemini provides embeddings and a long-context fallback. No agent framework — native Anthropic + Gemini + Qdrant SDKs, with LlamaIndex *readers only* for ingestion.

**Tech Stack:** Python 3.11+, uv, Hydra+OmegaConf, Qdrant, Anthropic SDK, google-generativeai, sentence-transformers (BGE reranker), Stim, mcp Python SDK, Streamlit, Click, slowapi, unstructured, defuddle, LlamaIndex readers (arxiv/github/youtube/file).

**Spec reference:** [`docs/superpowers/specs/2026-05-29-chatqec-design.md`](../specs/2026-05-29-chatqec-design.md)

---

## Milestone overview

| # | Milestone | Deliverable | Approx. tasks |
|---|---|---|---|
| 1 | Project scaffolding & repo bootstrap | `chatqec --help` runs; CI green; uv-managed project | 9 |
| 2 | Source registry & lifecycle | `chatqec sources list/show/mark-retracted/compact` works on a real JSONL registry | 7 |
| 3 | Canonical wiki layer | Frontmatter parser + 5 P0 pages written; validation passes | 8 |
| 4 | Ingestion pipeline | All 5 fetchers + math-aware chunker; `chatqec ingest --pdf/--arxiv/--github/--youtube/--url` produces chunks | 12 |
| 5 | Embedding + Qdrant storage | End-to-end: fixture corpus → embedded → searchable; Gemini default + local fallback | 8 |
| 6 | Retrieval pipeline | Hybrid retrieve + RRF + reranker + canonical force-injection; benchmarked on eval fixtures | 9 |
| 7 | Conversation & synthesis (no tools) | Multi-turn chat with Claude Sonnet + citations + Haiku rewriter/classifier; streaming via library API | 10 |
| 8 | MCP tool server + adapter | Sibling `chatqec-mcp-tools` repo with `stim_simulate`/`stim_diagram` in Docker sandbox; chat invokes tools end-to-end | 11 |
| 9 | Streamlit app + public-facing | 4-page app, rate limits, CAPTCHA, cost guard, off-topic refusal surfaced in UI | 9 |
| 10 | Eval set + watchlist automation | 50-pair Q&A eval set, metrics, weekly watchlist GitHub Action | 8 |

Each milestone ends with a green test suite and a commit point.

---

## Milestone 1 — Project scaffolding & repo bootstrap

**Outcome:** `chatqec --help` prints the CLI help, `pytest` shows ≥1 passing test, CI workflow runs green on push, `uv sync` installs a locked dependency tree, repo is connected to GitLab.

**Files this milestone creates / touches:**
- Create: `pyproject.toml`, `.python-version`, `uv.lock` (generated)
- Create: `README.md`, `LICENSE`, `CHANGELOG.md`
- Create: `conf/config.yaml`, `conf/__init__.py`
- Create: `src/chatqec/__init__.py`, `src/chatqec/config.py`, `src/chatqec/cli.py`, `src/chatqec/api.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_cli.py`, `tests/test_config.py`, `tests/test_api.py`
- Create: `.github/workflows/ci.yml`
- Create: `.pre-commit-config.yaml`
- Modify: `.gitignore` (already present; nothing to change in M1)

---

### Task 1.0 — Initialize git repository and connect to GitLab

**Files:**
- Modify: working directory (git init)

- [ ] **Step 1: Init local repo and stage the spec + .gitignore**

```bash
cd /home/pza/chatQEC
git init -b main
git add .gitignore docs/superpowers/specs/2026-05-29-chatqec-design.md
git commit -m "chore: bootstrap repo with design spec and .gitignore"
```

- [ ] **Step 2: Create the GitLab project (manual)**

Open https://gitlab.com/projects/new in a browser, create a new project named `chatqec`, choose **Private** initially (can be made public later), do NOT initialize with README (we already have files).

- [ ] **Step 3: Add the GitLab remote and push**

Replace `<USERNAME>` with your GitLab username:

```bash
git remote add origin git@gitlab.com:<USERNAME>/chatqec.git
git push -u origin main
```

Expected output: `Branch 'main' set up to track 'origin/main'.`

- [ ] **Step 4: Verify the push**

Open the GitLab project page in a browser; you should see the `.gitignore` and the spec under `docs/superpowers/specs/`.

---

### Task 1.1 — Author `pyproject.toml`

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`

- [ ] **Step 1: Pin Python version**

Create `.python-version`:

```
3.11
```

- [ ] **Step 2: Create `pyproject.toml` with project metadata, dependencies, and tool configs**

Create `pyproject.toml`:

```toml
[project]
name = "chatqec"
version = "0.1.0-dev"
description = "Semi-automated QEC research assistant with retrieval-augmented generation and sandboxed Stim execution."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
authors = [
  { name = "Sharmin Afrose", email = "sharmin.afrose.vt@gmail.com" },
]
keywords = ["quantum-error-correction", "qec", "rag", "llm", "anthropic", "gemini", "stim"]
classifiers = [
  "Development Status :: 2 - Pre-Alpha",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: Apache Software License",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Topic :: Scientific/Engineering :: Physics",
]

dependencies = [
  "anthropic>=0.40",
  "google-generativeai>=0.8",
  "qdrant-client>=1.12",
  "sentence-transformers>=3.0",
  "llama-index-readers-arxiv",
  "llama-index-readers-github",
  "llama-index-readers-youtube-transcript",
  "llama-index-readers-file",
  "defuddle",
  "unstructured[pdf]",
  "streamlit>=1.40",
  "click>=8.1",
  "hydra-core>=1.3",
  "omegaconf>=2.3",
  "mcp>=0.9",
  "pydantic>=2.8",
  "slowapi",
  "tenacity",
  "arxiv",
  "rich",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "pytest-asyncio>=0.24",
  "ruff>=0.6",
  "mypy>=1.11",
  "pre-commit>=3.7",
]

[project.scripts]
chatqec = "chatqec.cli:main"

[project.urls]
Homepage = "https://gitlab.com/<USERNAME>/chatqec"
Repository = "https://gitlab.com/<USERNAME>/chatqec"
Issues = "https://gitlab.com/<USERNAME>/chatqec/-/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/chatqec"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "PL", "RUF"]
ignore = ["E501", "PLR0913"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["PLR2004", "S101"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_unimported = true
show_error_codes = true
files = ["src"]

[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=chatqec --cov-report=term-missing"
filterwarnings = ["error"]

[tool.coverage.run]
branch = true
source = ["chatqec"]

[tool.coverage.report]
exclude_also = ["if TYPE_CHECKING:", "raise NotImplementedError"]
```

Replace `<USERNAME>` with your GitLab username.

---

### Task 1.2 — Install dependencies with uv and verify

**Files:**
- Generate: `uv.lock`
- Generate: `.venv/`

- [ ] **Step 1: Install uv (if not already installed)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If already installed, skip. Check with: `uv --version`.

- [ ] **Step 2: Create venv and install deps**

```bash
cd /home/pza/chatQEC
uv venv
uv sync --extra dev
```

Expected output (tail): `Installed N packages in X.XXs`.

- [ ] **Step 3: Verify Python and Click are importable**

```bash
uv run python -c "import click, anthropic, qdrant_client; print('ok')"
```

Expected output: `ok`

- [ ] **Step 4: Commit lockfile**

```bash
git add pyproject.toml .python-version uv.lock
git commit -m "chore: configure uv project with v1 dependency tree"
```

---

### Task 1.3 — Hydra config skeleton + loader test

**Files:**
- Create: `conf/__init__.py` (empty)
- Create: `conf/config.yaml`
- Create: `tests/test_config_loader.py`

- [ ] **Step 1: Write the failing test for config loading**

Create `tests/test_config_loader.py`:

```python
"""Smoke test that Hydra config composes and loads at startup."""

from hydra import compose, initialize_config_dir
from pathlib import Path


def test_default_config_loads():
    conf_dir = str(Path(__file__).resolve().parent.parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config")
    assert cfg.project.name == "chatqec"
    assert cfg.retriever.top_k == 8


def test_config_is_immutable_via_omegaconf():
    """Sanity: top-level project name is a string, retriever.top_k is an int."""
    conf_dir = str(Path(__file__).resolve().parent.parent / "conf")
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="config")
    assert isinstance(cfg.project.name, str)
    assert isinstance(cfg.retriever.top_k, int)
```

- [ ] **Step 2: Create empty `conf/__init__.py`**

```bash
touch conf/__init__.py
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_config_loader.py -v
```

Expected: FAIL — config file does not exist.

- [ ] **Step 4: Write `conf/config.yaml` to make the test pass**

Create `conf/config.yaml`:

```yaml
# ChatQEC root config (Hydra).
# Sub-configs live under conf/{retriever,synthesizer,ingestion}/ (added in later milestones).

project:
  name: chatqec
  version: "0.1.0-dev"

models:
  rewriter: claude-haiku-4-5
  classifier: claude-haiku-4-5
  synthesizer: claude-sonnet-4-6
  embedder: text-embedding-004
  long_context_fallback: gemini-2.5-pro
  reranker: BAAI/bge-reranker-base

retriever:
  top_k: 8
  candidate_k: 50
  alpha_vector: 0.5
  beta_bm25: 0.3
  gamma_tier: 0.15
  delta_recency: 0.05

paths:
  registry: sources/registry.jsonl
  watchlist: sources/watchlist.yaml
  canonical_dir: knowledge/canonical
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_config_loader.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add conf/ tests/test_config_loader.py
git commit -m "feat(config): add Hydra root config and loader smoke test"
```

---

### Task 1.4 — Frozen config dataclasses

**Files:**
- Create: `src/chatqec/__init__.py`
- Create: `src/chatqec/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for typed config**

Create `tests/test_config.py`:

```python
"""Frozen dataclass config wrappers around the Hydra config."""

import dataclasses
import pytest

from chatqec.config import RetrieverConfig, ModelsConfig, ProjectConfig, ChatQECConfig


def test_retriever_config_is_frozen():
    cfg = RetrieverConfig(top_k=8, candidate_k=50, alpha_vector=0.5,
                          beta_bm25=0.3, gamma_tier=0.15, delta_recency=0.05)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.top_k = 16  # type: ignore[misc]


def test_models_config_holds_model_ids():
    cfg = ModelsConfig(
        rewriter="claude-haiku-4-5",
        classifier="claude-haiku-4-5",
        synthesizer="claude-sonnet-4-6",
        embedder="text-embedding-004",
        long_context_fallback="gemini-2.5-pro",
        reranker="BAAI/bge-reranker-base",
    )
    assert cfg.synthesizer.startswith("claude-")
    assert cfg.embedder == "text-embedding-004"


def test_chatqec_config_aggregates_subconfigs():
    cfg = ChatQECConfig(
        project=ProjectConfig(name="chatqec", version="0.1.0-dev"),
        models=ModelsConfig(
            rewriter="claude-haiku-4-5",
            classifier="claude-haiku-4-5",
            synthesizer="claude-sonnet-4-6",
            embedder="text-embedding-004",
            long_context_fallback="gemini-2.5-pro",
            reranker="BAAI/bge-reranker-base",
        ),
        retriever=RetrieverConfig(top_k=8, candidate_k=50, alpha_vector=0.5,
                                  beta_bm25=0.3, gamma_tier=0.15, delta_recency=0.05),
    )
    assert cfg.project.name == "chatqec"
```

- [ ] **Step 2: Make `src/chatqec/` a package**

Create `src/chatqec/__init__.py`:

```python
"""ChatQEC — semi-automated QEC research assistant."""

__version__ = "0.1.0-dev"
__all__ = ["__version__"]
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_config.py -v
```

Expected: FAIL — `chatqec.config` module missing.

- [ ] **Step 4: Implement `src/chatqec/config.py` to make the test pass**

```python
"""Frozen dataclass mirrors of the Hydra config; safer for downstream code."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str


@dataclass(frozen=True)
class ModelsConfig:
    rewriter: str
    classifier: str
    synthesizer: str
    embedder: str
    long_context_fallback: str
    reranker: str


@dataclass(frozen=True)
class RetrieverConfig:
    top_k: int
    candidate_k: int
    alpha_vector: float
    beta_bm25: float
    gamma_tier: float
    delta_recency: float


@dataclass(frozen=True)
class ChatQECConfig:
    project: ProjectConfig
    models: ModelsConfig
    retriever: RetrieverConfig
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/__init__.py src/chatqec/config.py tests/test_config.py
git commit -m "feat(config): add frozen dataclass config wrappers"
```

---

### Task 1.5 — Click CLI shell with `chatqec --help` and `chatqec ask` stub

**Files:**
- Create: `src/chatqec/cli.py`
- Create: `src/chatqec/api.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing CLI test**

Create `tests/test_cli.py`:

```python
"""CLI shell smoke tests using Click's CliRunner."""

from click.testing import CliRunner

from chatqec.cli import main


def test_cli_help_includes_program_name():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "chatqec" in result.output.lower()


def test_cli_version_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_ask_subcommand_exists():
    runner = CliRunner()
    result = runner.invoke(main, ["ask", "--help"])
    assert result.exit_code == 0
    assert "question" in result.output.lower()


def test_cli_ask_with_stub_returns_placeholder():
    runner = CliRunner()
    result = runner.invoke(main, ["ask", "What is the surface code?"])
    assert result.exit_code == 0
    assert "stub" in result.output.lower() or "not yet implemented" in result.output.lower()
```

- [ ] **Step 2: Write failing API stub test**

Create `tests/test_api.py`:

```python
"""Public ChatQEC class — stub returning a placeholder until M7."""

from chatqec.api import ChatQEC, Answer


def test_chatqec_ask_returns_answer_object():
    bot = ChatQEC()
    answer = bot.ask("What is the surface code?")
    assert isinstance(answer, Answer)
    assert answer.text  # non-empty
    assert answer.citations == []
    assert answer.confidence in {"low", "medium", "high"}


def test_answer_is_immutable():
    bot = ChatQEC()
    answer = bot.ask("foo")
    # dataclass(frozen=True) raises on field assignment
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        answer.text = "changed"  # type: ignore[misc]
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py tests/test_api.py -v
```

Expected: 6 failures — modules don't exist.

- [ ] **Step 4: Implement `src/chatqec/api.py`**

```python
"""Public ChatQEC class. Stub in M1; wired up in M7."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Citation:
    source_id: str
    chunk_id: str
    title: str
    url: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, object]
    output: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    confidence: str = "low"


class ChatQEC:
    """Stub public API. Real wiring lands in Milestone 7."""

    def ask(self, question: str) -> Answer:
        return Answer(
            text=f"[stub] Asked: {question!r}. Synthesis not yet implemented (lands in M7).",
            citations=[],
            tool_calls=[],
            confidence="low",
        )
```

- [ ] **Step 5: Implement `src/chatqec/cli.py`**

```python
"""Click-based CLI entry point. Subcommands fill in across later milestones."""

from __future__ import annotations

import click

from chatqec import __version__
from chatqec.api import ChatQEC


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="chatqec")
def main() -> None:
    """ChatQEC — open-source QEC research assistant."""


@main.command()
@click.argument("question", nargs=-1, required=True)
def ask(question: tuple[str, ...]) -> None:
    """Answer a single QEC question (one-shot)."""
    bot = ChatQEC()
    answer = bot.ask(" ".join(question))
    click.echo(answer.text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py tests/test_api.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Smoke test the installed entry point**

```bash
uv run chatqec --help
uv run chatqec ask "What is the surface code?"
```

Expected: help output, then a stub answer.

- [ ] **Step 8: Commit**

```bash
git add src/chatqec/cli.py src/chatqec/api.py tests/test_cli.py tests/test_api.py
git commit -m "feat(cli): scaffold Click CLI and ChatQEC stub returning Answer objects"
```

---

### Task 1.6 — Pytest baseline, ruff and mypy configs

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Make `tests/` a package**

```bash
touch tests/__init__.py
```

- [ ] **Step 2: Create minimal `tests/conftest.py`**

```python
"""Pytest configuration for ChatQEC."""

import pytest


@pytest.fixture
def fixtures_dir():
    """Path to the tests/fixtures directory (created lazily in later milestones)."""
    from pathlib import Path
    return Path(__file__).parent / "fixtures"
```

- [ ] **Step 3: Run full pytest and lint**

```bash
uv run pytest -v
uv run ruff check src tests
uv run mypy src
```

Expected: all green. (If mypy complains about missing stubs for third-party libs in this scaffolding phase, add `disable_error_code = ["import-untyped"]` under `[tool.mypy]` and re-run.)

- [ ] **Step 4: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add conftest baseline and tests package"
```

---

### Task 1.7 — Pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write the pre-commit config**

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: detect-private-key
      - id: check-added-large-files
        args: ["--maxkb=500"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        files: ^src/
        additional_dependencies:
          - pydantic>=2.8
          - types-PyYAML
```

- [ ] **Step 2: Install pre-commit and run once across all files**

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Expected: hooks may auto-fix whitespace / EOL; rerun until clean.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit (ruff, mypy, hygiene hooks)"
```

---

### Task 1.8 — GitLab CI workflow

**Files:**
- Create: `.gitlab-ci.yml`

- [ ] **Step 1: Write the CI config (GitLab-native)**

Create `.gitlab-ci.yml`:

```yaml
stages:
  - lint
  - test

variables:
  UV_VERSION: "0.4.18"
  PYTHON_VERSION: "3.11"

.uv-base:
  image: python:3.11-slim
  before_script:
    - apt-get update -qq && apt-get install -y -qq curl git build-essential
    - curl -LsSf https://astral.sh/uv/install.sh | sh
    - export PATH="$HOME/.local/bin:$PATH"
    - uv --version
    - uv sync --extra dev
  cache:
    key:
      files: [uv.lock]
    paths:
      - .venv/

lint:
  extends: .uv-base
  stage: lint
  script:
    - uv run ruff check src tests
    - uv run mypy src

test:
  extends: .uv-base
  stage: test
  script:
    - uv run pytest -v --cov=chatqec --cov-report=term-missing --cov-report=xml
  coverage: '/(?i)total.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

- [ ] **Step 2: Validate YAML locally (optional but fast)**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.gitlab-ci.yml'))"
```

Expected: no output (no exception).

- [ ] **Step 3: Commit**

```bash
git add .gitlab-ci.yml
git commit -m "ci: add GitLab CI with lint and test stages"
```

---

### Task 1.9 — README, LICENSE, CHANGELOG

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `CHANGELOG.md`

- [ ] **Step 1: Write `LICENSE` (Apache 2.0 boilerplate)**

```bash
curl -sSL https://www.apache.org/licenses/LICENSE-2.0.txt -o LICENSE
```

Or if curl is blocked, paste the standard Apache 2.0 text manually into `LICENSE`.

- [ ] **Step 2: Write `README.md`**

Create `README.md`:

```markdown
# ChatQEC

Open-source semi-automated research assistant for Quantum Error Correction.

ChatQEC answers QEC questions over a curated corpus of research papers, textbooks,
tutorials, code repositories, and video transcripts, with inline citations and
sandboxed Stim execution for circuit-level questions.

## Status

Pre-alpha. See [the design spec](docs/superpowers/specs/2026-05-29-chatqec-design.md)
and [the implementation plan](docs/superpowers/plans/2026-05-29-chatqec-implementation.md).

## Quick start (developers)

```bash
uv venv
uv sync --extra dev
uv run chatqec --help
uv run pytest
```

## Stack

Python 3.11+, uv, Hydra, Qdrant, Anthropic + Gemini SDKs, Streamlit. Sandboxed
QEC tooling (Stim, PyMatching, qiskit-qec) lives in a sibling
[`chatqec-mcp-tools`](https://gitlab.com/<USERNAME>/chatqec-mcp-tools) MCP server.

## License

Apache 2.0 — see [LICENSE](LICENSE).
```

Replace `<USERNAME>` with your GitLab username.

- [ ] **Step 3: Write `CHANGELOG.md`**

Create `CHANGELOG.md`:

```markdown
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
```

- [ ] **Step 4: Commit and push the milestone**

```bash
git add README.md LICENSE CHANGELOG.md
git commit -m "docs: add README, LICENSE (Apache-2.0), and CHANGELOG"
git push origin main
```

Expected: GitLab shows all M1 commits.

---

## Milestone 1 — exit checklist

- [ ] `uv run chatqec --help` shows the CLI usage
- [ ] `uv run chatqec ask "test"` prints the stub answer
- [ ] `uv run pytest -v` passes (≥6 tests)
- [ ] `uv run ruff check src tests` passes
- [ ] `uv run mypy src` passes
- [ ] GitLab CI pipeline is green on `main`
- [ ] Repo visible on GitLab with all M1 commits

**🛑 PAUSE — push to GitLab and review before requesting Milestone 2.**

---
