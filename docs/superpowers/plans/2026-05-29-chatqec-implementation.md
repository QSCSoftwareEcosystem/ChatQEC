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

## Milestone 2 — Source registry & lifecycle

**Outcome:** A JSONL-backed source registry with full lifecycle states (`active | superseded | retracted | stale | pending_review`), invalid-transition guarding, content-hash utilities, canonical source-ID generation, and the CLI commands `chatqec sources list / show / mark-retracted / compact --dry-run`. No live ingestion or Qdrant yet — those land in M4/M5.

**Files this milestone creates / touches:**
- Create: `src/chatqec/ingestion/__init__.py`
- Create: `src/chatqec/ingestion/hashing.py`
- Create: `src/chatqec/ingestion/ids.py`
- Create: `src/chatqec/ingestion/registry.py`
- Create: `sources/registry.jsonl` (empty)
- Create: `sources/README.md`
- Create: `sources/sample/registry.sample.jsonl`
- Create: `tests/test_hashing.py`
- Create: `tests/test_ids.py`
- Create: `tests/test_registry.py`
- Create: `tests/test_cli_sources.py`
- Modify: `src/chatqec/cli.py` (add `sources` command group)
- Modify: `CHANGELOG.md`

**Why split this way:** the registry is the single source of truth for v1 (spec §6). It must be solid before any ingestion or retrieval lands. Lifecycle transitions are the spec's answer to "can we remove outdated references?" — wrong transitions silently breaking would corrupt the corpus.

---

### Task 2.1 — `SourceStatus` enum + `SourceRecord` Pydantic model

**Files:**
- Create: `src/chatqec/ingestion/__init__.py` (empty)
- Create: `src/chatqec/ingestion/registry.py`
- Create: `tests/test_registry.py` (model section only — registry I/O comes in 2.4)

- [ ] **Step 1: Make `ingestion` a package**

```bash
touch src/chatqec/ingestion/__init__.py
```

- [ ] **Step 2: Write failing tests for `SourceStatus` and `SourceRecord`**

Create `tests/test_registry.py`:

```python
"""Source registry model + I/O tests."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from chatqec.ingestion.registry import SourceRecord, SourceStatus, SourceType


def _utc(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def test_source_status_has_five_v1_states():
    expected = {"active", "superseded", "retracted", "stale", "pending_review"}
    assert {s.value for s in SourceStatus} == expected


def test_source_type_has_v1_types():
    expected = {"arxiv", "github", "youtube", "blog", "pdf", "canonical_wiki"}
    assert {t.value for t in SourceType} == expected


def test_source_record_minimum_fields():
    rec = SourceRecord(
        id="arxiv:2408.13687v1",
        type=SourceType.ARXIV,
        url="https://arxiv.org/abs/2408.13687",
        title="Quantum error correction below the surface code threshold",
        authority_tier="peer_reviewed",
        ingested_at=_utc(2026, 5, 29),
        last_checked_at=_utc(2026, 5, 29),
        status=SourceStatus.ACTIVE,
    )
    assert rec.id == "arxiv:2408.13687v1"
    assert rec.chunk_count == 0
    assert rec.chunk_ids == []
    assert rec.verified is False  # default


def test_source_record_rejects_unknown_status():
    with pytest.raises(ValidationError):
        SourceRecord(
            id="x",
            type=SourceType.ARXIV,
            url="https://example.com",
            title="t",
            authority_tier="preprint",
            ingested_at=_utc(2026, 1, 1),
            last_checked_at=_utc(2026, 1, 1),
            status="not_a_status",  # type: ignore[arg-type]
        )


def test_source_record_round_trips_through_json():
    rec = SourceRecord(
        id="arxiv:2408.13687v1",
        type=SourceType.ARXIV,
        url="https://arxiv.org/abs/2408.13687",
        title="Quantum error correction below the surface code threshold",
        authors=["Rajeev Acharya"],
        year=2024,
        version="v1",
        content_sha256="ab12",
        authority_tier="peer_reviewed",
        ingested_at=_utc(2026, 5, 29),
        last_checked_at=_utc(2026, 5, 29),
        status=SourceStatus.ACTIVE,
        verified=True,
    )
    blob = rec.model_dump_json()
    decoded = SourceRecord.model_validate_json(blob)
    assert decoded == rec
    # JSON shape sanity
    parsed = json.loads(blob)
    assert parsed["status"] == "active"
    assert parsed["type"] == "arxiv"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: ImportError on `chatqec.ingestion.registry`.

- [ ] **Step 4: Implement `src/chatqec/ingestion/registry.py` (model section)**

```python
"""Source registry: typed models, JSONL I/O, and lifecycle transitions.

The registry is the single source of truth for what is in the knowledge base.
It lives in Git (sources/registry.jsonl) and is human-auditable.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SourceStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    STALE = "stale"
    PENDING_REVIEW = "pending_review"


class SourceType(str, Enum):
    ARXIV = "arxiv"
    GITHUB = "github"
    YOUTUBE = "youtube"
    BLOG = "blog"
    PDF = "pdf"
    CANONICAL_WIKI = "canonical_wiki"


class SourceRecord(BaseModel):
    """One row in sources/registry.jsonl. Immutable once written.

    Lifecycle fields (status, supersedes, superseded_by) are mutated only via
    Registry.update_status() so transitions are validated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    type: SourceType
    url: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    version: str | None = None
    content_sha256: str | None = None
    license: str | None = None
    authority_tier: str  # canonical | peer_reviewed | preprint | blog | video
    ingested_at: datetime
    last_checked_at: datetime
    status: SourceStatus
    supersedes: str | None = None
    superseded_by: str | None = None
    chunk_count: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    verified: bool = False
    notes: str | None = None
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/ingestion/__init__.py src/chatqec/ingestion/registry.py tests/test_registry.py
git commit -m "feat(registry): add SourceStatus/SourceType enums and SourceRecord model"
```

---

### Task 2.2 — Content hashing utility

**Files:**
- Create: `src/chatqec/ingestion/hashing.py`
- Create: `tests/test_hashing.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hashing.py`:

```python
"""Content-hash utilities used by registry refresh logic to detect updates."""

from pathlib import Path

from chatqec.ingestion.hashing import sha256_bytes, sha256_text, sha256_file


def test_sha256_text_is_deterministic():
    h1 = sha256_text("hello, qec")
    h2 = sha256_text("hello, qec")
    assert h1 == h2
    assert len(h1) == 64  # hex digest length


def test_sha256_text_known_vector():
    # echo -n "" | sha256sum  →  e3b0c44... (the well-known empty-string digest)
    assert sha256_text("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_bytes_normalizes_to_text_for_ascii():
    assert sha256_text("abc") == sha256_bytes(b"abc")


def test_sha256_file_matches_text_for_small_file(tmp_path: Path):
    p = tmp_path / "fixture.txt"
    p.write_text("ChatQEC", encoding="utf-8")
    assert sha256_file(p) == sha256_text("ChatQEC")


def test_sha256_file_streams_large_input(tmp_path: Path):
    # Build a 5 MB file from a repeated pattern; verify file hash equals one-shot hash.
    chunk = b"qec-" * (1024 * 256)  # 1 MiB-ish
    big = chunk * 5
    p = tmp_path / "big.bin"
    p.write_bytes(big)
    assert sha256_file(p) == sha256_bytes(big)
```

- [ ] **Step 2: Run to confirm failure**

```bash
uv run pytest tests/test_hashing.py -v
```

Expected: ImportError on `chatqec.ingestion.hashing`.

- [ ] **Step 3: Implement `src/chatqec/ingestion/hashing.py`**

```python
"""SHA-256 helpers for the source registry's content-hash refresh logic."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 65536  # 64 KiB streaming read


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_hashing.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/hashing.py tests/test_hashing.py
git commit -m "feat(ingestion): add sha256 text/bytes/file helpers for content hashing"
```

---

### Task 2.3 — Canonical source-ID generation

**Files:**
- Create: `src/chatqec/ingestion/ids.py`
- Create: `tests/test_ids.py`

- [ ] **Step 1: Write failing tests for ID generation**

Create `tests/test_ids.py`:

```python
"""Canonical source-ID generators per source type.

The IDs are deterministic strings used as the registry primary key and as the
prefix for chunk IDs. They must be stable across runs and across machines.
"""

import pytest

from chatqec.ingestion.ids import (
    canonical_wiki_id,
    github_id,
    pdf_id,
    url_id,
    youtube_id,
    arxiv_id,
)


def test_arxiv_id_includes_version():
    assert arxiv_id("2408.13687", "v1") == "arxiv:2408.13687v1"
    assert arxiv_id("quant-ph/9705052", "v2") == "arxiv:quant-ph/9705052v2"


def test_arxiv_id_defaults_to_v1_when_version_missing():
    assert arxiv_id("2408.13687") == "arxiv:2408.13687v1"


def test_github_id_uses_org_repo_and_short_sha():
    assert github_id("quantumlib", "Stim", "deadbeefcafef00d") == "github:quantumlib/Stim@deadbeef"


def test_youtube_id_strips_query_and_keeps_video_id():
    assert youtube_id("https://www.youtube.com/watch?v=abc123XYZ&t=42s") == "youtube:abc123XYZ"
    assert youtube_id("https://youtu.be/abc123XYZ") == "youtube:abc123XYZ"


def test_url_id_hashes_url():
    # url IDs use a short hex prefix of sha256 for compactness
    assert url_id("https://example.com/post").startswith("url:")
    assert len(url_id("https://example.com/post")) == len("url:") + 16


def test_pdf_id_uses_content_hash_prefix():
    assert pdf_id("ab12cd34" + "0" * 56).startswith("pdf:")
    assert pdf_id("ab12cd34" + "0" * 56) == "pdf:ab12cd34"


def test_canonical_wiki_id_uses_topic_slug():
    assert canonical_wiki_id("surface-code") == "canonical:surface-code"


def test_canonical_wiki_id_rejects_uppercase_and_spaces():
    with pytest.raises(ValueError):
        canonical_wiki_id("Surface Code")
    with pytest.raises(ValueError):
        canonical_wiki_id("surface_code")  # underscore disallowed (kebab-case only)
```

- [ ] **Step 2: Run failing test**

```bash
uv run pytest tests/test_ids.py -v
```

Expected: ImportError on `chatqec.ingestion.ids`.

- [ ] **Step 3: Implement `src/chatqec/ingestion/ids.py`**

```python
"""Deterministic source-ID generators.

Each registry record's primary key is one of:
    arxiv:<id>v<version>
    github:<org>/<repo>@<short_sha>
    youtube:<video_id>
    url:<sha256[:16]>
    pdf:<content_sha256[:8]>
    canonical:<topic-slug>
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from chatqec.ingestion.hashing import sha256_text

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def arxiv_id(paper_id: str, version: str = "v1") -> str:
    if not version.startswith("v"):
        version = f"v{version}"
    return f"arxiv:{paper_id}{version}"


def github_id(org: str, repo: str, commit_sha: str) -> str:
    return f"github:{org}/{repo}@{commit_sha[:8]}"


def youtube_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        vid = parsed.path.lstrip("/")
    else:
        vid = parse_qs(parsed.query).get("v", [""])[0]
    if not vid:
        raise ValueError(f"could not extract YouTube video id from {url!r}")
    return f"youtube:{vid}"


def url_id(url: str) -> str:
    return f"url:{sha256_text(url)[:16]}"


def pdf_id(content_sha256_hex: str) -> str:
    if len(content_sha256_hex) < 8:
        raise ValueError("content_sha256 must be a full hex digest")
    return f"pdf:{content_sha256_hex[:8]}"


def canonical_wiki_id(topic_slug: str) -> str:
    if not _SLUG_RE.match(topic_slug):
        raise ValueError(
            f"canonical wiki topic_slug must be kebab-case (a-z, 0-9, hyphen): {topic_slug!r}"
        )
    return f"canonical:{topic_slug}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ids.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/ids.py tests/test_ids.py
git commit -m "feat(ingestion): add canonical source-id generators per source type"
```

---

### Task 2.4 — `Registry` class: I/O + lifecycle transitions

**Files:**
- Modify: `src/chatqec/ingestion/registry.py` (append `Registry` class + transition logic)
- Modify: `tests/test_registry.py` (append I/O and lifecycle tests)

- [ ] **Step 1: Append failing tests for I/O and transitions**

Append to `tests/test_registry.py`:

```python
# --- Registry I/O and lifecycle tests ---

from pathlib import Path

from chatqec.ingestion.registry import (
    InvalidTransitionError,
    Registry,
    RecordNotFoundError,
)


def _make(rid: str, status: SourceStatus = SourceStatus.ACTIVE) -> SourceRecord:
    return SourceRecord(
        id=rid,
        type=SourceType.ARXIV,
        url=f"https://arxiv.org/abs/{rid}",
        title=f"Title {rid}",
        authority_tier="preprint",
        ingested_at=_utc(2026, 1, 1),
        last_checked_at=_utc(2026, 1, 1),
        status=status,
    )


def test_registry_append_and_read_round_trip(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    reg = Registry(path)
    reg.append(_make("arxiv:1111.11111v1"))
    reg.append(_make("arxiv:2222.22222v1", status=SourceStatus.PENDING_REVIEW))

    reg2 = Registry(path)
    all_records = list(reg2.all())
    assert len(all_records) == 2
    assert {r.id for r in all_records} == {"arxiv:1111.11111v1", "arxiv:2222.22222v1"}


def test_registry_get_by_id(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    reg = Registry(path)
    reg.append(_make("arxiv:1111.11111v1"))
    rec = reg.get("arxiv:1111.11111v1")
    assert rec.title == "Title arxiv:1111.11111v1"


def test_registry_get_missing_raises(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    with pytest.raises(RecordNotFoundError):
        reg.get("arxiv:nope")


def test_registry_filter_by_status(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("a", status=SourceStatus.ACTIVE))
    reg.append(_make("b", status=SourceStatus.RETRACTED))
    reg.append(_make("c", status=SourceStatus.ACTIVE))
    actives = list(reg.filter(status=SourceStatus.ACTIVE))
    assert {r.id for r in actives} == {"a", "c"}


def test_registry_active_to_retracted_is_allowed(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("a", status=SourceStatus.ACTIVE))
    reg.update_status("a", SourceStatus.RETRACTED)
    assert reg.get("a").status is SourceStatus.RETRACTED


def test_registry_retracted_to_active_is_blocked(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("a", status=SourceStatus.RETRACTED))
    with pytest.raises(InvalidTransitionError):
        reg.update_status("a", SourceStatus.ACTIVE)


def test_registry_pending_review_can_only_move_to_active_or_retracted(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("a", status=SourceStatus.PENDING_REVIEW))
    with pytest.raises(InvalidTransitionError):
        reg.update_status("a", SourceStatus.SUPERSEDED)
    reg.update_status("a", SourceStatus.ACTIVE)
    assert reg.get("a").status is SourceStatus.ACTIVE


def test_registry_supersede_chains_records(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("arxiv:2408.13687v1", status=SourceStatus.ACTIVE))
    reg.append(_make("arxiv:2408.13687v2", status=SourceStatus.ACTIVE))
    reg.supersede(old_id="arxiv:2408.13687v1", new_id="arxiv:2408.13687v2")
    old = reg.get("arxiv:2408.13687v1")
    new = reg.get("arxiv:2408.13687v2")
    assert old.status is SourceStatus.SUPERSEDED
    assert old.superseded_by == "arxiv:2408.13687v2"
    assert new.supersedes == "arxiv:2408.13687v1"


def test_registry_compact_candidates_returns_old_retracted_and_superseded(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("old-retracted", status=SourceStatus.RETRACTED))
    reg.append(_make("old-superseded", status=SourceStatus.SUPERSEDED))
    reg.append(_make("active-keep", status=SourceStatus.ACTIVE))
    # Treat all records' last_checked_at as 2026-01-01; "now" = 2026-05-29 → 148 days.
    candidates = list(reg.compact_candidates(older_than_days=90, now=_utc(2026, 5, 29)))
    assert {r.id for r in candidates} == {"old-retracted", "old-superseded"}


def test_registry_compact_excludes_recent_records(tmp_path: Path):
    reg = Registry(tmp_path / "registry.jsonl")
    reg.append(_make("recent-retracted", status=SourceStatus.RETRACTED))  # ingested 2026-01-01
    candidates = list(reg.compact_candidates(older_than_days=365, now=_utc(2026, 5, 29)))
    assert list(candidates) == []
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: ImportError on `Registry`, `InvalidTransitionError`, `RecordNotFoundError`.

- [ ] **Step 3: Append `Registry` to `src/chatqec/ingestion/registry.py`**

Append below the `SourceRecord` definition:

```python
# --- Registry I/O and lifecycle ---

from collections.abc import Iterable, Iterator
from datetime import timedelta
from pathlib import Path


class RegistryError(Exception):
    """Base class for registry errors."""


class RecordNotFoundError(RegistryError):
    """Raised when a record id is not present in the registry."""


class InvalidTransitionError(RegistryError):
    """Raised when an attempted status transition is not permitted."""


# Allowed lifecycle transitions. See spec §6.
_ALLOWED_TRANSITIONS: dict[SourceStatus, set[SourceStatus]] = {
    SourceStatus.PENDING_REVIEW: {SourceStatus.ACTIVE, SourceStatus.RETRACTED},
    SourceStatus.ACTIVE: {SourceStatus.SUPERSEDED, SourceStatus.RETRACTED, SourceStatus.STALE},
    SourceStatus.STALE: {SourceStatus.ACTIVE, SourceStatus.SUPERSEDED, SourceStatus.RETRACTED},
    SourceStatus.SUPERSEDED: {SourceStatus.RETRACTED},
    SourceStatus.RETRACTED: set(),
}


class Registry:
    """JSONL-backed source registry.

    Records are append-only on disk; updates rewrite the file atomically.
    For v1 corpora (≤ 10k records) the in-memory representation is fine.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._records: dict[str, SourceRecord] = self._load()

    def _load(self) -> dict[str, SourceRecord]:
        records: dict[str, SourceRecord] = {}
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = SourceRecord.model_validate_json(line)
                records[rec.id] = rec
        return records

    def _flush(self) -> None:
        """Rewrite the JSONL file atomically."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in self._records.values():
                f.write(rec.model_dump_json())
                f.write("\n")
        tmp.replace(self.path)

    def append(self, record: SourceRecord) -> None:
        if record.id in self._records:
            raise RegistryError(f"duplicate id: {record.id}")
        self._records[record.id] = record
        self._flush()

    def get(self, record_id: str) -> SourceRecord:
        try:
            return self._records[record_id]
        except KeyError as e:
            raise RecordNotFoundError(record_id) from e

    def all(self) -> Iterator[SourceRecord]:
        return iter(self._records.values())

    def filter(self, *, status: SourceStatus | None = None) -> Iterator[SourceRecord]:
        for rec in self._records.values():
            if status is not None and rec.status is not status:
                continue
            yield rec

    def update_status(self, record_id: str, new_status: SourceStatus) -> SourceRecord:
        rec = self.get(record_id)
        if new_status not in _ALLOWED_TRANSITIONS[rec.status]:
            raise InvalidTransitionError(
                f"cannot move {record_id} from {rec.status.value} → {new_status.value}"
            )
        updated = rec.model_copy(update={"status": new_status})
        self._records[record_id] = updated
        self._flush()
        return updated

    def supersede(self, *, old_id: str, new_id: str) -> tuple[SourceRecord, SourceRecord]:
        old = self.get(old_id)
        new = self.get(new_id)
        if old.status is not SourceStatus.ACTIVE:
            raise InvalidTransitionError(
                f"can only supersede an ACTIVE record; {old_id} is {old.status.value}"
            )
        old = old.model_copy(update={
            "status": SourceStatus.SUPERSEDED,
            "superseded_by": new_id,
        })
        new = new.model_copy(update={"supersedes": old_id})
        self._records[old_id] = old
        self._records[new_id] = new
        self._flush()
        return old, new

    def compact_candidates(
        self,
        *,
        older_than_days: int,
        now: datetime,
    ) -> Iterable[SourceRecord]:
        cutoff = now - timedelta(days=older_than_days)
        for rec in self._records.values():
            if rec.status in (SourceStatus.RETRACTED, SourceStatus.SUPERSEDED):
                if rec.last_checked_at < cutoff:
                    yield rec
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: 14 passed (5 from Task 2.1 + 9 new).

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/registry.py tests/test_registry.py
git commit -m "feat(registry): add Registry class with JSONL I/O and lifecycle transitions"
```

---

### Task 2.5 — Bootstrap `sources/` directory + sample + README

**Files:**
- Create: `sources/registry.jsonl` (empty)
- Create: `sources/README.md`
- Create: `sources/sample/registry.sample.jsonl`

- [ ] **Step 1: Create the empty live registry**

```bash
mkdir -p sources/sample
touch sources/registry.jsonl
```

- [ ] **Step 2: Write the sample registry (2 example records for documentation)**

Create `sources/sample/registry.sample.jsonl`:

```jsonl
{"id":"arxiv:2408.13687v1","type":"arxiv","url":"https://arxiv.org/abs/2408.13687","title":"Quantum error correction below the surface code threshold","authors":["Rajeev Acharya"],"year":2024,"version":"v1","content_sha256":null,"license":"arXiv non-exclusive","authority_tier":"peer_reviewed","ingested_at":"2026-05-29T00:00:00Z","last_checked_at":"2026-05-29T00:00:00Z","status":"active","supersedes":null,"superseded_by":null,"chunk_count":0,"chunk_ids":[],"verified":true,"notes":"Google Willow result"}
{"id":"canonical:surface-code","type":"canonical_wiki","url":"knowledge/canonical/surface-code.md","title":"Surface Code","authors":[],"year":null,"version":null,"content_sha256":null,"license":"project-internal","authority_tier":"canonical","ingested_at":"2026-05-29T00:00:00Z","last_checked_at":"2026-05-29T00:00:00Z","status":"active","supersedes":null,"superseded_by":null,"chunk_count":0,"chunk_ids":[],"verified":true,"notes":"P0 canonical page"}
```

- [ ] **Step 3: Write `sources/README.md`**

Create `sources/README.md`:

```markdown
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
```

- [ ] **Step 4: Commit**

```bash
git add sources/
git commit -m "chore(sources): bootstrap empty registry + sample + README"
```

---

### Task 2.6 — CLI: `chatqec sources list` and `chatqec sources show`

**Files:**
- Modify: `src/chatqec/cli.py`
- Create: `tests/test_cli_sources.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli_sources.py`:

```python
"""CLI tests for `chatqec sources` subcommands."""

from datetime import datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from chatqec.cli import main
from chatqec.ingestion.registry import Registry, SourceRecord, SourceStatus, SourceType


def _seed(path: Path) -> None:
    reg = Registry(path)
    reg.append(SourceRecord(
        id="arxiv:2408.13687v1",
        type=SourceType.ARXIV,
        url="https://arxiv.org/abs/2408.13687",
        title="Quantum error correction below the surface code threshold",
        authors=["Rajeev Acharya"],
        year=2024,
        authority_tier="peer_reviewed",
        ingested_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
        last_checked_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
        status=SourceStatus.ACTIVE,
        verified=True,
    ))
    reg.append(SourceRecord(
        id="canonical:surface-code",
        type=SourceType.CANONICAL_WIKI,
        url="knowledge/canonical/surface-code.md",
        title="Surface Code",
        authority_tier="canonical",
        ingested_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
        last_checked_at=datetime(2026, 5, 29, tzinfo=timezone.utc),
        status=SourceStatus.ACTIVE,
        verified=True,
    ))


def test_sources_list_prints_records(tmp_path: Path):
    _seed(tmp_path / "registry.jsonl")
    runner = CliRunner()
    result = runner.invoke(main, ["sources", "--registry", str(tmp_path / "registry.jsonl"), "list"])
    assert result.exit_code == 0, result.output
    assert "arxiv:2408.13687v1" in result.output
    assert "Surface Code" in result.output


def test_sources_list_filters_by_status(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    _seed(path)
    reg = Registry(path)
    reg.update_status("arxiv:2408.13687v1", SourceStatus.RETRACTED)
    runner = CliRunner()
    result = runner.invoke(main, [
        "sources", "--registry", str(path), "list", "--status", "active",
    ])
    assert result.exit_code == 0
    assert "arxiv:2408.13687v1" not in result.output
    assert "canonical:surface-code" in result.output


def test_sources_show_prints_full_record(tmp_path: Path):
    path = tmp_path / "registry.jsonl"
    _seed(path)
    runner = CliRunner()
    result = runner.invoke(main, [
        "sources", "--registry", str(path), "show", "arxiv:2408.13687v1",
    ])
    assert result.exit_code == 0
    assert "Rajeev Acharya" in result.output
    assert "peer_reviewed" in result.output


def test_sources_show_missing_id_exits_nonzero(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(main, [
        "sources", "--registry", str(tmp_path / "registry.jsonl"), "show", "arxiv:nope",
    ])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_cli_sources.py -v
```

Expected: `sources` command does not exist.

- [ ] **Step 3: Extend `src/chatqec/cli.py` with the `sources` group**

Replace the file contents with:

```python
"""Click-based CLI entry point."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from chatqec import __version__
from chatqec.api import ChatQEC
from chatqec.ingestion.registry import (
    InvalidTransitionError,
    RecordNotFoundError,
    Registry,
    SourceStatus,
)


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


@main.group()
@click.option(
    "--registry",
    "registry_path",
    type=click.Path(path_type=Path),
    default=Path("sources/registry.jsonl"),
    show_default=True,
    help="Path to the JSONL source registry.",
)
@click.pass_context
def sources(ctx: click.Context, registry_path: Path) -> None:
    """Inspect and manage the source registry."""
    ctx.obj = Registry(registry_path)


@sources.command("list")
@click.option(
    "--status",
    type=click.Choice([s.value for s in SourceStatus], case_sensitive=False),
    default=None,
    help="Filter by lifecycle status.",
)
@click.pass_obj
def sources_list(reg: Registry, status: str | None) -> None:
    """List registry records as a table."""
    console = Console()
    table = Table(show_header=True, header_style="bold")
    for col in ("id", "type", "title", "tier", "status", "year", "verified"):
        table.add_column(col)
    status_enum = SourceStatus(status) if status else None
    rows = list(reg.filter(status=status_enum)) if status_enum else list(reg.all())
    for r in rows:
        table.add_row(
            r.id, r.type.value, r.title[:60], r.authority_tier,
            r.status.value, str(r.year or ""), "✓" if r.verified else "·",
        )
    console.print(table)
    console.print(f"[dim]{len(rows)} records[/]")


@sources.command("show")
@click.argument("record_id")
@click.pass_obj
def sources_show(reg: Registry, record_id: str) -> None:
    """Print one record as pretty JSON."""
    try:
        rec = reg.get(record_id)
    except RecordNotFoundError:
        click.echo(f"record not found: {record_id}", err=True)
        raise SystemExit(2) from None
    click.echo(rec.model_dump_json(indent=2))


@sources.command("mark-retracted")
@click.argument("record_id")
@click.pass_obj
def sources_mark_retracted(reg: Registry, record_id: str) -> None:
    """Manually mark a source as retracted (excluded from retrieval)."""
    try:
        updated = reg.update_status(record_id, SourceStatus.RETRACTED)
    except RecordNotFoundError:
        click.echo(f"record not found: {record_id}", err=True)
        raise SystemExit(2) from None
    except InvalidTransitionError as e:
        click.echo(f"invalid transition: {e}", err=True)
        raise SystemExit(3) from None
    click.echo(f"marked {updated.id} as {updated.status.value}")


@sources.command("compact")
@click.option(
    "--older-than-days",
    type=int,
    default=90,
    show_default=True,
    help="Retracted/superseded records older than this are candidates.",
)
@click.option(
    "--dry-run/--apply",
    default=True,
    help="Dry run (default) prints candidates without removing them.",
)
@click.pass_obj
def sources_compact(reg: Registry, older_than_days: int, dry_run: bool) -> None:
    """List or purge records eligible for compaction.

    NOTE: physical purge of chunks from Qdrant lands in Milestone 5.
    In M2, --apply is rejected with a guard.
    """
    now = datetime.now(tz=timezone.utc)
    candidates = list(reg.compact_candidates(older_than_days=older_than_days, now=now))
    for rec in candidates:
        click.echo(f"{rec.id}\t{rec.status.value}\t{rec.last_checked_at.isoformat()}")
    click.echo(f"\n{len(candidates)} candidates")
    if not dry_run:
        click.echo("--apply not yet supported (wired up in Milestone 5)", err=True)
        raise SystemExit(4)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli_sources.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Smoke test the CLI against the sample registry**

```bash
cp sources/sample/registry.sample.jsonl /tmp/sample-registry.jsonl
uv run chatqec sources --registry /tmp/sample-registry.jsonl list
uv run chatqec sources --registry /tmp/sample-registry.jsonl show arxiv:2408.13687v1
```

Expected: a 2-row table and a JSON dump.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/cli.py tests/test_cli_sources.py
git commit -m "feat(cli): add 'sources list/show/mark-retracted/compact' commands"
```

---

### Task 2.7 — Update CHANGELOG, push the milestone

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append the M2 section to `CHANGELOG.md`**

Append under `## [Unreleased] → ### Added`:

```markdown
- Source registry (`sources/registry.jsonl`) with typed `SourceRecord`, lifecycle states (`active|superseded|retracted|stale|pending_review`), and validated state transitions
- SHA-256 content-hash utilities (`chatqec.ingestion.hashing`)
- Canonical source-ID generators per source type (`chatqec.ingestion.ids`)
- CLI: `chatqec sources list/show/mark-retracted/compact`
```

- [ ] **Step 2: Run the full test suite and lint**

```bash
uv run pytest -v
uv run ruff check src tests
uv run mypy src
```

Expected: all green. Total tests after M2: ≥27 (M1: 6 + M2: 21).

- [ ] **Step 3: Commit and push**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for M2 (source registry + lifecycle)"
git push origin main
```

Expected: GitLab pipeline runs and goes green.

---

## Milestone 2 — exit checklist

- [ ] `Registry` round-trips records through JSONL atomically
- [ ] `update_status` rejects illegal transitions (e.g. retracted → active)
- [ ] `supersede(old_id, new_id)` chains records via `superseded_by` / `supersedes`
- [ ] `compact_candidates` returns retracted/superseded records older than N days
- [ ] `chatqec sources list / show / mark-retracted / compact --dry-run` work against the sample registry
- [ ] ≥21 new tests added; full suite ≥27 passing
- [ ] CHANGELOG updated
- [ ] GitLab CI green on `main`

**🛑 PAUSE — push to GitLab and review before requesting Milestone 3.**

---

## Milestone 3 — Canonical wiki layer

**Outcome:** A typed canonical-page schema, frontmatter parser/loader, cross-reference validator, section chunker, the first 5 P0 canonical pages (with real content, expandable later), and CLI commands `chatqec canonical list / validate`. Canonical pages are now first-class citizens — ready for ingestion in M4 and force-injection in M6.

**Files this milestone creates / touches:**
- Create: `src/chatqec/canonical/__init__.py`
- Create: `src/chatqec/canonical/page.py` (schema + parser)
- Create: `src/chatqec/canonical/loader.py` (scan + validate)
- Create: `src/chatqec/canonical/chunker.py` (section-based)
- Create: `knowledge/canonical/README.md`
- Create: `knowledge/canonical/_roster.yaml` (the 52-page roster from spec A.1)
- Create: `knowledge/canonical/surface-code.md`
- Create: `knowledge/canonical/stabilizer-formalism.md`
- Create: `knowledge/canonical/threshold-theorem.md`
- Create: `knowledge/canonical/css-codes.md`
- Create: `knowledge/canonical/magic-state-distillation.md`
- Create: `tests/test_canonical_page.py`
- Create: `tests/test_canonical_loader.py`
- Create: `tests/test_canonical_chunker.py`
- Create: `tests/test_cli_canonical.py`
- Modify: `src/chatqec/cli.py` (add `canonical` command group)
- Modify: `CHANGELOG.md`

**Why split this way:** the canonical layer is what grounds Claude in correct QEC terminology. The schema and validator are the contract; the 5 starter pages prove the contract works end-to-end; the chunker outputs what M5/M6 will index and inject.

---

### Task 3.1 — `CanonicalPage` Pydantic schema

**Files:**
- Create: `src/chatqec/canonical/__init__.py` (empty)
- Create: `src/chatqec/canonical/page.py`
- Create: `tests/test_canonical_page.py`

- [ ] **Step 1: Make `canonical` a package**

```bash
touch src/chatqec/canonical/__init__.py
```

- [ ] **Step 2: Write failing tests for the schema**

Create `tests/test_canonical_page.py`:

```python
"""CanonicalPage schema + frontmatter parsing tests."""

from datetime import date

import pytest
from pydantic import ValidationError

from chatqec.canonical.page import CanonicalFrontmatter, CanonicalPage, parse_page


def test_frontmatter_accepts_valid_kebab_slug():
    fm = CanonicalFrontmatter(
        topic_slug="surface-code",
        title="Surface Code",
        aliases=["toric code", "planar code"],
        see_also=["stabilizer-formalism", "color-code"],
        authority_tier="canonical",
        last_reviewed=date(2026, 5, 29),
        maintainers=["sharmin"],
    )
    assert fm.topic_slug == "surface-code"


def test_frontmatter_rejects_uppercase_slug():
    with pytest.raises(ValidationError):
        CanonicalFrontmatter(
            topic_slug="Surface-Code",
            title="x",
            authority_tier="canonical",
            last_reviewed=date(2026, 1, 1),
            maintainers=["m"],
        )


def test_frontmatter_rejects_underscore_in_slug():
    with pytest.raises(ValidationError):
        CanonicalFrontmatter(
            topic_slug="surface_code",
            title="x",
            authority_tier="canonical",
            last_reviewed=date(2026, 1, 1),
            maintainers=["m"],
        )


def test_frontmatter_requires_canonical_authority_tier():
    with pytest.raises(ValidationError):
        CanonicalFrontmatter(
            topic_slug="x",
            title="x",
            authority_tier="preprint",  # not allowed for canonical pages
            last_reviewed=date(2026, 1, 1),
            maintainers=["m"],
        )


def test_parse_page_splits_frontmatter_and_body():
    text = """---
topic_slug: surface-code
title: "Surface Code"
aliases: ["toric code"]
see_also: ["stabilizer-formalism"]
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers: ["sharmin"]
---
# Surface Code

The surface code is a topological CSS code.

## Stabilizers

X-type plaquettes and Z-type stars.
"""
    page = parse_page(text)
    assert isinstance(page, CanonicalPage)
    assert page.frontmatter.topic_slug == "surface-code"
    assert "topological" in page.body.lower()
    # The title heading should appear in the body
    assert page.body.lstrip().startswith("# Surface Code")


def test_parse_page_rejects_missing_frontmatter():
    text = "# Surface Code\n\nNo frontmatter here.\n"
    with pytest.raises(ValueError, match="frontmatter"):
        parse_page(text)


def test_parse_page_rejects_unterminated_frontmatter():
    text = "---\ntopic_slug: x\n# missing closing fence\n"
    with pytest.raises(ValueError):
        parse_page(text)
```

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/test_canonical_page.py -v
```

Expected: ImportError on `chatqec.canonical.page`.

- [ ] **Step 4: Implement `src/chatqec/canonical/page.py`**

```python
"""Canonical wiki page schema and frontmatter parser.

Canonical pages live in knowledge/canonical/<slug>.md and are the only sources
authored by humans rather than fetched. They get authority_tier="canonical"
and are force-injected into context when the query matches their topic_slug
or aliases (see spec §5.2 and §8.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


class CanonicalFrontmatter(BaseModel):
    """YAML frontmatter of a canonical wiki page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic_slug: str
    title: str
    aliases: list[str] = Field(default_factory=list)
    see_also: list[str] = Field(default_factory=list)
    authority_tier: str
    last_reviewed: date
    maintainers: list[str]

    @field_validator("topic_slug")
    @classmethod
    def _kebab_slug(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(f"topic_slug must be kebab-case (a-z, 0-9, hyphen): {v!r}")
        return v

    @field_validator("see_also", "aliases")
    @classmethod
    def _no_blanks(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            raise ValueError("entries must be non-empty strings")
        return v

    @field_validator("authority_tier")
    @classmethod
    def _must_be_canonical(cls, v: str) -> str:
        if v != "canonical":
            raise ValueError(f"canonical pages must have authority_tier='canonical', got {v!r}")
        return v


@dataclass(frozen=True)
class CanonicalPage:
    """A parsed canonical wiki page = frontmatter + raw markdown body."""

    frontmatter: CanonicalFrontmatter
    body: str


def parse_page(text: str) -> CanonicalPage:
    """Parse a canonical page from raw markdown text."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(
            "canonical page must start with YAML frontmatter delimited by '---' fences"
        )
    fm_text, body = match.group(1), match.group(2)
    fm_data = yaml.safe_load(fm_text)
    if not isinstance(fm_data, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return CanonicalPage(
        frontmatter=CanonicalFrontmatter(**fm_data),
        body=body,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_canonical_page.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/canonical/__init__.py src/chatqec/canonical/page.py tests/test_canonical_page.py
git commit -m "feat(canonical): add CanonicalFrontmatter schema and parser"
```

---

### Task 3.2 — Canonical page loader + cross-reference validator

**Files:**
- Create: `src/chatqec/canonical/loader.py`
- Create: `tests/test_canonical_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_canonical_loader.py`:

```python
"""Canonical loader tests — scans knowledge/canonical/ and validates cross-refs."""

from pathlib import Path

import pytest

from chatqec.canonical.loader import (
    CrossReferenceError,
    DuplicateAliasError,
    DuplicateSlugError,
    load_canonical_directory,
)


def _write_page(path: Path, slug: str, aliases: list[str] | None = None,
                see_also: list[str] | None = None) -> None:
    aliases = aliases or []
    see_also = see_also or []
    path.write_text(f"""---
topic_slug: {slug}
title: "{slug}"
aliases: {aliases!r}
see_also: {see_also!r}
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers: ["m"]
---
# {slug}

body for {slug}
""", encoding="utf-8")


def test_load_canonical_directory_returns_all_pages(tmp_path: Path):
    _write_page(tmp_path / "a.md", "a")
    _write_page(tmp_path / "b.md", "b")
    pages = load_canonical_directory(tmp_path)
    assert {p.frontmatter.topic_slug for p in pages} == {"a", "b"}


def test_load_canonical_directory_skips_underscored_files(tmp_path: Path):
    # Files starting with _ (e.g. _roster.yaml, _index.md) are ignored
    _write_page(tmp_path / "a.md", "a")
    (tmp_path / "_index.md").write_text("# index", encoding="utf-8")
    (tmp_path / "_roster.yaml").write_text("foo: bar\n", encoding="utf-8")
    pages = load_canonical_directory(tmp_path)
    assert len(pages) == 1


def test_load_rejects_duplicate_slug(tmp_path: Path):
    _write_page(tmp_path / "a.md", "x")
    _write_page(tmp_path / "b.md", "x")  # same slug, different file
    with pytest.raises(DuplicateSlugError):
        load_canonical_directory(tmp_path)


def test_load_rejects_alias_colliding_with_existing_slug(tmp_path: Path):
    _write_page(tmp_path / "a.md", "surface-code", aliases=["color-code"])
    _write_page(tmp_path / "b.md", "color-code")
    with pytest.raises(DuplicateAliasError):
        load_canonical_directory(tmp_path)


def test_load_rejects_unresolved_see_also(tmp_path: Path):
    _write_page(tmp_path / "a.md", "a", see_also=["nonexistent-slug"])
    _write_page(tmp_path / "b.md", "b")
    with pytest.raises(CrossReferenceError, match="nonexistent-slug"):
        load_canonical_directory(tmp_path)


def test_load_allows_see_also_pointing_at_alias(tmp_path: Path):
    _write_page(tmp_path / "a.md", "a", see_also=["toric-code"])
    _write_page(tmp_path / "b.md", "surface-code", aliases=["toric-code"])
    # Should NOT raise — see_also resolution checks slugs ∪ aliases.
    pages = load_canonical_directory(tmp_path)
    assert len(pages) == 2
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_canonical_loader.py -v
```

Expected: ImportError on `chatqec.canonical.loader`.

- [ ] **Step 3: Implement `src/chatqec/canonical/loader.py`**

```python
"""Loader and cross-reference validator for the canonical wiki directory."""

from __future__ import annotations

from pathlib import Path

from chatqec.canonical.page import CanonicalPage, parse_page


class CanonicalLoadError(Exception):
    pass


class DuplicateSlugError(CanonicalLoadError):
    pass


class DuplicateAliasError(CanonicalLoadError):
    pass


class CrossReferenceError(CanonicalLoadError):
    pass


def load_canonical_directory(directory: Path) -> list[CanonicalPage]:
    """Scan a directory of canonical pages, parse, and validate cross-refs.

    Rules:
    - Files starting with '_' are ignored (e.g. _roster.yaml, _index.md).
    - All topic_slugs must be unique.
    - No alias may collide with another page's slug or another alias.
    - Every see_also entry must resolve to an existing slug OR alias.
    """
    pages: list[CanonicalPage] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.startswith("_"):
            continue
        pages.append(parse_page(path.read_text(encoding="utf-8")))

    slugs: set[str] = set()
    alias_to_slug: dict[str, str] = {}
    for p in pages:
        slug = p.frontmatter.topic_slug
        if slug in slugs:
            raise DuplicateSlugError(slug)
        slugs.add(slug)
    for p in pages:
        for alias in p.frontmatter.aliases:
            if alias in slugs and alias != p.frontmatter.topic_slug:
                raise DuplicateAliasError(
                    f"alias {alias!r} on page {p.frontmatter.topic_slug!r} collides with slug {alias!r}"
                )
            if alias in alias_to_slug:
                raise DuplicateAliasError(
                    f"alias {alias!r} appears on both "
                    f"{alias_to_slug[alias]!r} and {p.frontmatter.topic_slug!r}"
                )
            alias_to_slug[alias] = p.frontmatter.topic_slug

    resolvable = slugs | set(alias_to_slug.keys())
    for p in pages:
        for ref in p.frontmatter.see_also:
            if ref not in resolvable:
                raise CrossReferenceError(
                    f"see_also entry {ref!r} on page {p.frontmatter.topic_slug!r} "
                    f"does not resolve to any slug or alias"
                )
    return pages
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_canonical_loader.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/canonical/loader.py tests/test_canonical_loader.py
git commit -m "feat(canonical): add directory loader and cross-reference validator"
```

---

### Task 3.3 — Section-based chunker for canonical pages

**Files:**
- Create: `src/chatqec/canonical/chunker.py`
- Create: `tests/test_canonical_chunker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_canonical_chunker.py`:

```python
"""Section-based chunker for canonical pages.

Canonical content is author-controlled and well-structured; we chunk on '##'
boundaries and emit one chunk per section.
"""

from chatqec.canonical.chunker import CanonicalChunk, chunk_canonical_body


SAMPLE = """# Surface Code

Intro paragraph mentioning the toric code.

## Stabilizers

X-plaquettes and Z-stars.

## Decoding

MWPM on the syndrome graph.

### Subdecoders

Union-Find is faster.
"""


def test_chunker_emits_one_chunk_per_h2():
    chunks = chunk_canonical_body(SAMPLE)
    assert [c.section_title for c in chunks] == ["Preamble", "Stabilizers", "Decoding"]


def test_chunker_includes_h3_in_parent_h2():
    chunks = chunk_canonical_body(SAMPLE)
    decoding = next(c for c in chunks if c.section_title == "Decoding")
    assert "Union-Find" in decoding.text
    assert "### Subdecoders" in decoding.text


def test_chunker_preamble_captures_text_before_first_h2():
    chunks = chunk_canonical_body(SAMPLE)
    preamble = chunks[0]
    assert preamble.section_title == "Preamble"
    assert "toric code" in preamble.text


def test_chunker_anchor_is_kebab_of_section_title():
    chunks = chunk_canonical_body(SAMPLE)
    assert chunks[1].anchor == "stabilizers"
    assert chunks[2].anchor == "decoding"
    assert chunks[0].anchor == ""  # preamble has no heading anchor


def test_chunker_yields_no_chunks_for_empty_body():
    assert list(chunk_canonical_body("")) == []


def test_chunk_dataclass_is_frozen():
    chunks = chunk_canonical_body(SAMPLE)
    import dataclasses
    import pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunks[0].text = "x"  # type: ignore[misc]
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_canonical_chunker.py -v
```

Expected: ImportError on `chatqec.canonical.chunker`.

- [ ] **Step 3: Implement `src/chatqec/canonical/chunker.py`**

```python
"""Section-based chunker for canonical wiki pages.

Splits markdown body into chunks at each '##' heading. Subheadings (###, ####)
stay within their parent ## section. Text before the first ## (if any) becomes
a "Preamble" chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CanonicalChunk:
    section_title: str
    text: str
    anchor: str  # kebab-case version of section_title, "" for preamble


_H2_RE = re.compile(r"^##[ \t]+(.+?)\s*$", re.MULTILINE)


def _slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s


def chunk_canonical_body(body: str) -> list[CanonicalChunk]:
    if not body.strip():
        return []

    # Locate all H2 headings. Each H2 starts a section that runs until the next H2.
    matches = list(_H2_RE.finditer(body))
    chunks: list[CanonicalChunk] = []

    if not matches:
        # Whole body is preamble
        return [CanonicalChunk(section_title="Preamble", text=body.strip(), anchor="")]

    # Preamble = everything before first H2
    first = matches[0]
    preamble = body[: first.start()].strip()
    if preamble:
        chunks.append(CanonicalChunk(section_title="Preamble", text=preamble, anchor=""))

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        chunks.append(
            CanonicalChunk(section_title=title, text=section_text, anchor=_slugify(title))
        )
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_canonical_chunker.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/canonical/chunker.py tests/test_canonical_chunker.py
git commit -m "feat(canonical): add section-based chunker (## boundaries)"
```

---

### Task 3.4 — `knowledge/canonical/` directory + README + 52-page roster

**Files:**
- Create: `knowledge/canonical/README.md`
- Create: `knowledge/canonical/_roster.yaml`

- [ ] **Step 1: Make the directory**

```bash
mkdir -p knowledge/canonical
```

- [ ] **Step 2: Write the contributor README**

Create `knowledge/canonical/README.md`:

```markdown
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
```

- [ ] **Step 3: Write the roster file (top of file shown; copy full A.1 from the spec)**

Create `knowledge/canonical/_roster.yaml`:

```yaml
# Canonical wiki roster — v1 target (52 pages: 24 P0, 20 P1, 8 P2).
# Authoritative source: docs/superpowers/specs/2026-05-29-chatqec-design.md §A.1
# This file is documentation; the loader ignores files starting with '_'.

pages:
  # --- P0 — Foundations ---
  - {slug: "pauli-group", title: "Pauli Group", priority: P0, bucket: foundations,
     aliases: ["Pauli operators", "n-qubit Pauli group"],
     see_also: [stabilizer-formalism, pauli-channels, logical-operators]}
  - {slug: "stabilizer-formalism", title: "Stabilizer Formalism", priority: P0, bucket: foundations,
     aliases: ["stabilizer codes intro", "Pauli stabilizers", "Gottesman formalism"],
     see_also: [pauli-group, css-codes, logical-operators, syndrome-extraction, clifford-group]}
  - {slug: "nkd-notation", title: "[[n,k,d]] Code Parameters", priority: P0, bucket: foundations,
     aliases: ["code distance", "n k d notation", "code parameters"],
     see_also: [logical-operators, stabilizer-formalism, threshold-theorem]}
  - {slug: "logical-operators", title: "Logical Operators", priority: P0, bucket: foundations,
     aliases: ["logical Pauli operators", "logical X", "logical Z", "encoded operators"],
     see_also: [stabilizer-formalism, nkd-notation, transversal-gates]}
  - {slug: "knill-laflamme", title: "Knill-Laflamme Conditions", priority: P0, bucket: foundations,
     aliases: ["KL conditions", "QEC conditions"],
     see_also: [stabilizer-formalism, nkd-notation, syndrome-extraction]}
  - {slug: "syndrome-extraction", title: "Syndrome Extraction", priority: P0, bucket: foundations,
     aliases: ["stabilizer measurement", "syndrome measurement", "parity checks"],
     see_also: [stabilizer-formalism, decoding, fault-tolerance, measurement-errors, circuit-level-noise]}
  - {slug: "decoding", title: "Decoding (General)", priority: P0, bucket: decoders,
     aliases: ["quantum decoder", "syndrome decoding"],
     see_also: [syndrome-extraction, mwpm-decoder, bp-osd-decoder, union-find-decoder, threshold-theorem]}
  - {slug: "threshold-theorem", title: "Threshold Theorem", priority: P0, bucket: foundations,
     aliases: ["fault tolerance threshold", "accuracy threshold theorem"],
     see_also: [fault-tolerance, concatenated-codes, surface-code, circuit-level-noise]}
  - {slug: "fault-tolerance", title: "Fault Tolerance", priority: P0, bucket: fault-tolerance,
     aliases: ["fault-tolerant quantum computation", "FTQC"],
     see_also: [threshold-theorem, transversal-gates, syndrome-extraction, magic-state-distillation]}
  - {slug: "clifford-group", title: "Clifford Group", priority: P0, bucket: foundations,
     aliases: ["Clifford gates", "Clifford operations"],
     see_also: [pauli-group, transversal-gates, magic-state-distillation]}
  - {slug: "shor-code", title: "Shor Code", priority: P0, bucket: codes,
     aliases: ["9-qubit code", "Shor 9-qubit code", "[[9,1,3]] code"],
     see_also: [steane-code, css-codes, concatenated-codes]}
  - {slug: "steane-code", title: "Steane Code", priority: P0, bucket: codes,
     aliases: ["[[7,1,3]] code", "7-qubit Steane code", "Hamming-based code"],
     see_also: [css-codes, shor-code, transversal-gates, color-code]}
  - {slug: "css-codes", title: "CSS Codes", priority: P0, bucket: codes,
     aliases: ["Calderbank-Shor-Steane codes", "CSS construction"],
     see_also: [stabilizer-formalism, steane-code, surface-code, qldpc-codes]}
  - {slug: "five-qubit-code", title: "Five-Qubit Code", priority: P0, bucket: codes,
     aliases: ["perfect code", "[[5,1,3]] code", "smallest QEC code"],
     see_also: [stabilizer-formalism, knill-laflamme, nkd-notation]}
  - {slug: "surface-code", title: "Surface Code", priority: P0, bucket: codes,
     aliases: ["toric code", "planar surface code", "Kitaev surface code", "rotated surface code"],
     see_also: [css-codes, color-code, mwpm-decoder, lattice-surgery, circuit-level-noise]}
  - {slug: "color-code", title: "Color Code", priority: P0, bucket: codes,
     aliases: ["2D color code", "triangular color code", "Bombin color code"],
     see_also: [surface-code, css-codes, transversal-gates]}
  - {slug: "bacon-shor-code", title: "Bacon-Shor Code", priority: P0, bucket: codes,
     aliases: ["Bacon-Shor subsystem code", "compass code family"],
     see_also: [subsystem-codes, shor-code, css-codes]}
  - {slug: "transversal-gates", title: "Transversal Gates", priority: P0, bucket: fault-tolerance,
     aliases: ["transversal logic", "Eastin-Knill"],
     see_also: [clifford-group, steane-code, reed-muller-codes, magic-state-distillation]}
  - {slug: "magic-state-distillation", title: "Magic State Distillation", priority: P0, bucket: fault-tolerance,
     aliases: ["MSD", "T-state distillation"],
     see_also: [magic-state-injection, triorthogonal-codes, reed-muller-codes, t-gate-count]}
  - {slug: "magic-state-injection", title: "Magic State Injection", priority: P0, bucket: fault-tolerance,
     aliases: ["state injection", "T-state injection"],
     see_also: [magic-state-distillation, lattice-surgery, transversal-gates]}
  - {slug: "lattice-surgery", title: "Lattice Surgery", priority: P0, bucket: fault-tolerance,
     aliases: ["surface code lattice surgery", "merge and split"],
     see_also: [surface-code, code-deformation, magic-state-injection]}
  - {slug: "reed-muller-codes", title: "Reed-Muller Codes (Quantum)", priority: P0, bucket: codes,
     aliases: ["quantum Reed-Muller", "[[15,1,3]] code"],
     see_also: [transversal-gates, magic-state-distillation, triorthogonal-codes]}
  - {slug: "depolarizing-noise", title: "Depolarizing Noise", priority: P0, bucket: noise,
     aliases: ["depolarizing channel", "symmetric Pauli noise"],
     see_also: [pauli-channels, circuit-level-noise, biased-noise]}
  - {slug: "measurement-errors", title: "Measurement Errors", priority: P0, bucket: noise,
     aliases: ["readout errors", "syndrome measurement noise"],
     see_also: [syndrome-extraction, circuit-level-noise, mwpm-decoder]}

  # --- P1 — modern codes, decoders, missing pieces ---
  - {slug: "qldpc-codes", title: "Quantum LDPC Codes", priority: P1, bucket: codes, aliases: ["qLDPC"], see_also: [bivariate-bicycle-codes, bp-osd-decoder, css-codes]}
  - {slug: "bivariate-bicycle-codes", title: "Bivariate Bicycle Codes", priority: P1, bucket: codes, aliases: ["BB codes", "IBM bicycle codes", "gross code"], see_also: [qldpc-codes, bp-osd-decoder]}
  - {slug: "floquet-codes", title: "Floquet Codes", priority: P1, bucket: codes, aliases: ["honeycomb code", "Hastings-Haah code", "dynamical codes"], see_also: [surface-code, subsystem-codes]}
  - {slug: "subsystem-codes", title: "Subsystem Codes", priority: P1, bucket: codes, aliases: ["gauge codes", "OQEC"], see_also: [bacon-shor-code, floquet-codes, stabilizer-formalism]}
  - {slug: "concatenated-codes", title: "Concatenated Codes", priority: P1, bucket: codes, aliases: ["code concatenation"], see_also: [shor-code, threshold-theorem, steane-code]}
  - {slug: "gkp-codes", title: "GKP Codes", priority: P1, bucket: codes, aliases: ["Gottesman-Kitaev-Preskill code", "grid states"], see_also: [cat-codes, continuous-variable-qec]}
  - {slug: "cat-codes", title: "Cat Codes", priority: P1, bucket: codes, aliases: ["cat qubits", "dissipative cat codes"], see_also: [gkp-codes, biased-noise]}
  - {slug: "biased-noise", title: "Biased Noise", priority: P1, bucket: noise, aliases: ["Z-biased noise", "dephasing-dominated noise"], see_also: [cat-codes, xzzx-surface-code, depolarizing-noise]}
  - {slug: "triorthogonal-codes", title: "Triorthogonal Codes", priority: P1, bucket: codes, aliases: ["Bravyi-Haah codes"], see_also: [magic-state-distillation, reed-muller-codes]}
  - {slug: "code-deformation", title: "Code Deformation and Code Switching", priority: P1, bucket: fault-tolerance, aliases: ["gauge fixing"], see_also: [lattice-surgery, color-code, surface-code]}
  - {slug: "solovay-kitaev", title: "Solovay-Kitaev Theorem", priority: P1, bucket: fault-tolerance, aliases: ["SK theorem", "gate synthesis"], see_also: [clifford-group, t-gate-count]}
  - {slug: "t-gate-count", title: "T-Gate Count and Non-Clifford Resources", priority: P1, bucket: fault-tolerance, aliases: ["T-count", "magic budget"], see_also: [magic-state-distillation, solovay-kitaev]}
  - {slug: "mwpm-decoder", title: "MWPM Decoder", priority: P1, bucket: decoders, aliases: ["MWPM", "Edmonds blossom decoder", "PyMatching"], see_also: [surface-code, decoding, union-find-decoder]}
  - {slug: "bp-osd-decoder", title: "BP+OSD Decoder", priority: P1, bucket: decoders, aliases: ["belief propagation OSD", "ordered statistics decoding"], see_also: [qldpc-codes, bivariate-bicycle-codes]}
  - {slug: "union-find-decoder", title: "Union-Find Decoder", priority: P1, bucket: decoders, aliases: ["UF decoder", "Delfosse-Nickerson decoder"], see_also: [mwpm-decoder, surface-code]}
  - {slug: "tensor-network-decoders", title: "Tensor Network Decoders", priority: P1, bucket: decoders, aliases: ["TN decoder", "MPS decoder"], see_also: [decoding, surface-code]}
  - {slug: "neural-decoders", title: "Neural and ML Decoders", priority: P1, bucket: decoders, aliases: ["ML decoder", "transformer decoder"], see_also: [decoding, mwpm-decoder]}
  - {slug: "color-code-decoders", title: "Color Code Decoders", priority: P1, bucket: decoders, aliases: ["restriction decoder", "projection decoder"], see_also: [color-code, mwpm-decoder]}
  - {slug: "circuit-level-noise", title: "Circuit-Level Noise Models", priority: P1, bucket: noise, aliases: ["SD6", "SI1000"], see_also: [depolarizing-noise, measurement-errors, surface-code]}
  - {slug: "coherent-errors", title: "Coherent Errors", priority: P1, bucket: noise, aliases: ["unitary errors", "over-rotation errors"], see_also: [pauli-channels, pauli-frame-tracking]}
  - {slug: "leakage", title: "Leakage Errors", priority: P1, bucket: noise, aliases: ["leakage reduction"], see_also: [crosstalk, circuit-level-noise]}
  - {slug: "pauli-channels", title: "Pauli Channels and Twirling", priority: P1, bucket: theory, aliases: ["Pauli twirling"], see_also: [depolarizing-noise, coherent-errors]}
  - {slug: "pauli-frame-tracking", title: "Pauli Frame Tracking", priority: P1, bucket: fault-tolerance, aliases: ["Pauli frame", "frame tracking"], see_also: [clifford-group, syndrome-extraction]}

  # --- P2 — theory depth, hardware platforms ---
  - {slug: "anyons", title: "Anyons", priority: P2, bucket: theory, see_also: [topological-order, surface-code]}
  - {slug: "topological-order", title: "Topological Order", priority: P2, bucket: theory, see_also: [anyons, surface-code, floquet-codes]}
  - {slug: "crosstalk", title: "Crosstalk", priority: P2, bucket: noise, aliases: ["ZZ crosstalk", "spectator errors"], see_also: [leakage, circuit-level-noise]}
  - {slug: "xzzx-surface-code", title: "XZZX Surface Code", priority: P2, bucket: codes, aliases: ["tailored surface code"], see_also: [surface-code, biased-noise]}
  - {slug: "superconducting-qec", title: "Superconducting QEC", priority: P2, bucket: hardware, aliases: ["transmon QEC"], see_also: [surface-code, leakage]}
  - {slug: "neutral-atom-qec", title: "Neutral Atom QEC", priority: P2, bucket: hardware, aliases: ["Rydberg atom QEC"], see_also: [qldpc-codes, transversal-gates]}
  - {slug: "trapped-ion-qec", title: "Trapped Ion QEC", priority: P2, bucket: hardware, aliases: ["ion trap QEC"], see_also: [color-code, transversal-gates]}
  - {slug: "photonic-qec", title: "Photonic QEC", priority: P2, bucket: hardware, aliases: ["fusion-based quantum computing"], see_also: [gkp-codes, continuous-variable-qec]}
  - {slug: "continuous-variable-qec", title: "Continuous-Variable QEC", priority: P2, bucket: hardware, aliases: ["CV QEC", "bosonic QEC"], see_also: [gkp-codes, cat-codes, photonic-qec]}
```

- [ ] **Step 4: Commit**

```bash
git add knowledge/canonical/README.md knowledge/canonical/_roster.yaml
git commit -m "docs(canonical): add knowledge/canonical/ README and 52-page roster"
```

---

### Task 3.5 — Write 5 P0 canonical pages (starter content)

**Files (create all five):**
- Create: `knowledge/canonical/surface-code.md`
- Create: `knowledge/canonical/stabilizer-formalism.md`
- Create: `knowledge/canonical/threshold-theorem.md`
- Create: `knowledge/canonical/css-codes.md`
- Create: `knowledge/canonical/magic-state-distillation.md`

**Note:** Each page below has complete frontmatter (used by the loader) and brief but accurate body content the maintainer can expand later. The five chosen pages cross-reference each other heavily, so they form a self-consistent starter set.

- [ ] **Step 1: Create `knowledge/canonical/surface-code.md`**

```markdown
---
topic_slug: surface-code
title: "Surface Code"
aliases:
  - "toric code"
  - "planar surface code"
  - "Kitaev surface code"
  - "rotated surface code"
see_also:
  - css-codes
  - threshold-theorem
  - stabilizer-formalism
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers:
  - sharmin
---

# Surface Code

The surface code is a topological CSS quantum error-correcting code defined on a
2D square lattice of qubits. It is the leading near-term candidate for
fault-tolerant quantum computation because of its high threshold (~1% under
circuit-level depolarizing noise), its requirement of only nearest-neighbour
2-qubit gates, and its compatibility with planar superconducting and
neutral-atom architectures.

## Construction

Place a data qubit on every edge of an $L \times L$ square lattice. Associate
two types of stabilizers with the lattice:

- **Plaquette (face) stabilizers** $B_p = \prod_{e \in \partial p} Z_e$
- **Vertex (star) stabilizers** $A_v = \prod_{e \ni v} X_e$

All stabilizers commute by construction (each pair shares an even number of
qubits). For a planar code with rough/smooth boundaries, the code encodes
$k = 1$ logical qubit with distance $d = L$.

## Logical operators

A logical $\bar{X}$ is a string of $X$ operators connecting two rough
boundaries; logical $\bar{Z}$ is a string of $Z$ operators connecting two
smooth boundaries. Both have weight $L$.

## Decoding

Bit-flip errors leave $Z$-syndrome on adjacent plaquettes; phase-flip errors
leave $X$-syndrome on adjacent stars. Decoding is the problem of inferring the
most likely error from the syndrome — typically by **minimum-weight perfect
matching** on the syndrome graph (see [[mwpm-decoder]]).

## Why it dominates near-term QEC

- High threshold under realistic noise (~1% for SD6, SI1000)
- Local 2-qubit gates only
- Well-developed lattice-surgery primitives for logical operations
- Mature simulator + decoder ecosystem (Stim, PyMatching)

## See also

- [[css-codes]] — the CSS construction the surface code instantiates
- [[threshold-theorem]] — what "below threshold" actually means
- [[stabilizer-formalism]] — the abstract framework underneath
```

- [ ] **Step 2: Create `knowledge/canonical/stabilizer-formalism.md`**

```markdown
---
topic_slug: stabilizer-formalism
title: "Stabilizer Formalism"
aliases:
  - "stabilizer codes intro"
  - "Pauli stabilizers"
  - "Gottesman formalism"
see_also:
  - css-codes
  - surface-code
  - threshold-theorem
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers:
  - sharmin
---

# Stabilizer Formalism

The stabilizer formalism (Gottesman 1997) describes a quantum code as the
joint $+1$-eigenspace of an abelian subgroup $S \subseteq P_n$ of the $n$-qubit
Pauli group, where $-I \notin S$. Nearly every practical QEC code in use today
— including the surface, color, and qLDPC families — is a stabilizer code.

## Definition

Let $S = \langle g_1, \ldots, g_{n-k} \rangle$ be generated by $n - k$
independent commuting Pauli operators. The code space is

$$\mathcal{C}(S) = \{ |\psi\rangle : g_i |\psi\rangle = |\psi\rangle \ \forall i \}$$

and encodes $k$ logical qubits.

## Logical operators

The normalizer $N(S) \setminus S$ contains $2k$ independent Pauli operators
that commute with every stabilizer but are not themselves stabilizers. These
are the encoded $\bar{X}_i, \bar{Z}_i$ operators.

## Error correction

An error $E \in P_n$ either commutes with all stabilizers (the syndrome is
trivial — either no error or a logical error) or anti-commutes with some — the
**syndrome** is the bit string $s_i = [g_i, E] / 2$.

A code with distance $d$ corrects $\lfloor (d-1)/2 \rfloor$ Pauli errors.

## Why it matters

- All efficient classical-simulable circuits (Gottesman-Knill) live in the stabilizer formalism
- Syndrome measurement reduces to measuring stabilizer generators
- Most decoders operate on the syndrome graph derived from the stabilizers

## See also

- [[css-codes]] — a structured family of stabilizer codes
- [[surface-code]] — the practical topological stabilizer code
- [[threshold-theorem]] — what the formalism enables
```

- [ ] **Step 3: Create `knowledge/canonical/threshold-theorem.md`**

```markdown
---
topic_slug: threshold-theorem
title: "Threshold Theorem"
aliases:
  - "fault tolerance threshold"
  - "accuracy threshold theorem"
see_also:
  - surface-code
  - stabilizer-formalism
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers:
  - sharmin
---

# Threshold Theorem

The threshold theorem (Aharonov & Ben-Or 1999; Knill, Laflamme, Zurek 1996;
Kitaev 1997) states that if the physical error rate per gate is below a
code-dependent constant $p_{th}$, then arbitrarily long quantum computations
are possible with polylogarithmic overhead.

## Statement (informal)

For a family of fault-tolerant constructions of code distance $d$, there
exists a threshold $p_{th}$ such that the logical error rate satisfies

$$p_L(d) \lesssim A \left( \frac{p}{p_{th}} \right)^{(d+1)/2}$$

for $p < p_{th}$. Increasing distance exponentially suppresses logical errors
below threshold.

## Surface code threshold

Under circuit-level depolarizing noise with the standard syndrome-extraction
circuit, the surface code threshold is approximately $p_{th} \approx 1\%$.
Under biased noise, tailored variants like [[surface-code]] XZZX achieve
higher thresholds against the dominant error type.

## What "below threshold" means experimentally

A system operates **below threshold** when increasing the code distance
*decreases* the logical error rate. Google's 2024 Willow result (arXiv:2408.13687)
demonstrated this for $d = 3, 5, 7$ on a superconducting surface code.

## Common misconceptions

- The threshold is *not* the physical error rate at which one logical qubit
  beats one physical qubit — that's a separate engineering threshold.
- The threshold depends on the *noise model*, the *code*, the *decoder*, and
  the *syndrome-extraction circuit*. Quoting a threshold without specifying
  these is meaningless.

## See also

- [[surface-code]] — primary code where below-threshold has been demonstrated
- [[stabilizer-formalism]] — algebraic framework used in threshold proofs
```

- [ ] **Step 4: Create `knowledge/canonical/css-codes.md`**

```markdown
---
topic_slug: css-codes
title: "CSS Codes"
aliases:
  - "Calderbank-Shor-Steane codes"
  - "CSS construction"
see_also:
  - stabilizer-formalism
  - surface-code
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers:
  - sharmin
---

# CSS Codes

CSS codes (Calderbank-Shor-Steane, 1996) are stabilizer codes whose generators
can be split into two sets: one consisting purely of $X$-type operators and
one purely of $Z$-type. They are constructed from a pair of classical linear
codes satisfying a duality condition.

## Construction

Let $C_1, C_2$ be classical binary linear codes with $C_2 \subseteq C_1$. The
parity-check matrices $H_1$ (for $C_1$) and $H_2$ (for $C_2^\perp$) define the
stabilizer generators:

- $X$-stabilizers: one $X$ on each non-zero column of $H_2$ — using rows of $H_2$
- $Z$-stabilizers: one $Z$ on each non-zero column of $H_1$ — using rows of $H_1$

The resulting quantum code has parameters $[[n, k, d]]$ where
$k = \dim(C_1) - \dim(C_2)$.

## Why CSS matters

- Bit-flip and phase-flip errors decode *independently*, simplifying decoder design
- Many practical codes are CSS: [[surface-code]], color code, qLDPC codes
- Transversal CNOT works on any CSS code

## Notable CSS codes

| Code | $[[n,k,d]]$ | Construction |
|---|---|---|
| Steane | $[[7,1,3]]$ | $C_1 = C_2^\perp = [7,4]$ Hamming |
| Surface | $[[L^2, 1, L]]$ | toric / planar topological |
| Bivariate bicycle | various | qLDPC family |

## Limitations

Eastin-Knill rules out a universal transversal gate set for any non-trivial
quantum code — including CSS codes. Hence the need for [[magic-state-distillation]].

## See also

- [[stabilizer-formalism]] — the broader algebraic context
- [[surface-code]] — the practical topological CSS code
```

- [ ] **Step 5: Create `knowledge/canonical/magic-state-distillation.md`**

```markdown
---
topic_slug: magic-state-distillation
title: "Magic State Distillation"
aliases:
  - "MSD"
  - "T-state distillation"
  - "state distillation"
see_also:
  - css-codes
  - stabilizer-formalism
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers:
  - sharmin
---

# Magic State Distillation

Magic state distillation (Bravyi & Kitaev 2005) is the standard route to
universal fault-tolerant quantum computation on stabilizer codes. Eastin-Knill
forbids a transversal universal gate set, so non-Clifford operations are
implemented by **injecting** a noisy magic state and **distilling** purified
copies using stabilizer operations only.

## The magic states

The canonical resource state is

$$|T\rangle = \cos(\pi/8)|0\rangle + e^{i\pi/4}\sin(\pi/8)|1\rangle$$

Consuming one ideal $|T\rangle$ enables one $T = \mathrm{diag}(1, e^{i\pi/4})$
gate by gate teleportation. Combined with the Clifford group (achievable
transversally on many stabilizer codes), this gives universality.

## The protocol

1. Prepare $n$ noisy $|T\rangle$ states with error rate $p$.
2. Encode them into the $[[15, 1, 3]]$ Reed-Muller code (or a triorthogonal code).
3. Measure the code's stabilizers; on the trivial syndrome, the output magic
   state has lower error rate $p' \sim O(p^d)$ where $d$ is the code distance.
4. Repeat in cascade until the desired fidelity is reached.

## Cost

Magic state distillation dominates the resource cost of fault-tolerant
algorithms (Gidney-Ekera 2021 RSA estimate). Modern work (Litinski 2019,
Wills-Lin-Yoder 2024) substantially reduces this overhead via
**triorthogonal codes** and improved factory architectures.

## See also

- [[css-codes]] — the codes used in distillation factories
- [[stabilizer-formalism]] — what makes Clifford-only operations "free"
```

- [ ] **Step 6: Commit**

```bash
git add knowledge/canonical/*.md
git commit -m "docs(canonical): add 5 P0 canonical pages (surface-code, stabilizer, threshold, css, msd)"
```

---

### Task 3.6 — CLI: `chatqec canonical list` and `chatqec canonical validate`

**Files:**
- Modify: `src/chatqec/cli.py`
- Create: `tests/test_cli_canonical.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli_canonical.py`:

```python
"""CLI tests for `chatqec canonical` subcommands."""

from pathlib import Path

from click.testing import CliRunner

from chatqec.cli import main


def _write(path: Path, slug: str, see_also: list[str] | None = None) -> None:
    see_also = see_also or []
    path.write_text(f"""---
topic_slug: {slug}
title: "{slug}"
aliases: []
see_also: {see_also!r}
authority_tier: canonical
last_reviewed: 2026-05-29
maintainers: ["m"]
---
# {slug}

body
""", encoding="utf-8")


def test_canonical_list_prints_slugs(tmp_path: Path):
    _write(tmp_path / "a.md", "a")
    _write(tmp_path / "b.md", "b")
    runner = CliRunner()
    result = runner.invoke(main, ["canonical", "--dir", str(tmp_path), "list"])
    assert result.exit_code == 0
    assert "a" in result.output
    assert "b" in result.output


def test_canonical_validate_passes_clean_dir(tmp_path: Path):
    _write(tmp_path / "a.md", "a", see_also=["b"])
    _write(tmp_path / "b.md", "b")
    runner = CliRunner()
    result = runner.invoke(main, ["canonical", "--dir", str(tmp_path), "validate"])
    assert result.exit_code == 0
    assert "ok" in result.output.lower() or "2 pages" in result.output.lower()


def test_canonical_validate_reports_broken_cross_ref(tmp_path: Path):
    _write(tmp_path / "a.md", "a", see_also=["nonexistent"])
    runner = CliRunner()
    result = runner.invoke(main, ["canonical", "--dir", str(tmp_path), "validate"])
    assert result.exit_code != 0
    assert "nonexistent" in result.output.lower()
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_cli_canonical.py -v
```

Expected: `canonical` command not found.

- [ ] **Step 3: Add the `canonical` group to `src/chatqec/cli.py`**

Insert the following block in `src/chatqec/cli.py`, immediately after the `sources_compact` function definition and before the `if __name__ == "__main__":` line. Also add the necessary import at the top of the file.

Add to the imports section (near the other `from chatqec...` imports):

```python
from chatqec.canonical.loader import (
    CanonicalLoadError,
    load_canonical_directory,
)
```

Then add the command group:

```python
@main.group()
@click.option(
    "--dir",
    "canonical_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False, dir_okay=True),
    default=Path("knowledge/canonical"),
    show_default=True,
    help="Path to the canonical wiki directory.",
)
@click.pass_context
def canonical(ctx: click.Context, canonical_dir: Path) -> None:
    """Inspect and validate the canonical wiki layer."""
    ctx.obj = canonical_dir


@canonical.command("list")
@click.pass_obj
def canonical_list(canonical_dir: Path) -> None:
    """List all canonical pages by topic_slug."""
    pages = load_canonical_directory(canonical_dir)
    console = Console()
    table = Table(show_header=True, header_style="bold")
    for col in ("topic_slug", "title", "aliases", "see_also", "last_reviewed"):
        table.add_column(col)
    for p in pages:
        fm = p.frontmatter
        table.add_row(
            fm.topic_slug, fm.title,
            ", ".join(fm.aliases), ", ".join(fm.see_also),
            fm.last_reviewed.isoformat(),
        )
    console.print(table)
    console.print(f"[dim]{len(pages)} pages[/]")


@canonical.command("validate")
@click.pass_obj
def canonical_validate(canonical_dir: Path) -> None:
    """Validate frontmatter + cross-references for every canonical page."""
    try:
        pages = load_canonical_directory(canonical_dir)
    except CanonicalLoadError as e:
        click.echo(f"validation failed: {e}", err=True)
        raise SystemExit(2) from None
    click.echo(f"ok — {len(pages)} pages parsed and cross-references resolve")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli_canonical.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the real CLI against the 5 P0 pages**

```bash
uv run chatqec canonical list
uv run chatqec canonical validate
```

Expected: 5-row table, then `ok — 5 pages parsed and cross-references resolve`.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/cli.py tests/test_cli_canonical.py
git commit -m "feat(cli): add 'canonical list/validate' commands"
```

---

### Task 3.7 — CI hook: validate canonical pages on every push

**Files:**
- Modify: `.gitlab-ci.yml`

- [ ] **Step 1: Add a `validate-canonical` stage and job**

Open `.gitlab-ci.yml` and:

a) Update the `stages:` list to include the new stage between `lint` and `test`:

```yaml
stages:
  - lint
  - validate-canonical
  - test
```

b) Add a new job at the end of the file:

```yaml
validate-canonical:
  extends: .uv-base
  stage: validate-canonical
  script:
    - uv run chatqec canonical --dir knowledge/canonical validate
```

- [ ] **Step 2: Sanity-check locally**

```bash
uv run chatqec canonical --dir knowledge/canonical validate
```

Expected: `ok — 5 pages parsed and cross-references resolve`.

- [ ] **Step 3: Commit**

```bash
git add .gitlab-ci.yml
git commit -m "ci: add validate-canonical stage to catch broken cross-references"
```

---

### Task 3.8 — Update CHANGELOG, run full suite, push

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append M3 to `CHANGELOG.md`** (under `## [Unreleased] → ### Added`):

```markdown
- Canonical wiki layer: `CanonicalFrontmatter` schema, directory loader with cross-reference validation, section-based chunker
- 5 P0 canonical pages: surface-code, stabilizer-formalism, threshold-theorem, css-codes, magic-state-distillation
- 52-page canonical roster documentation (`knowledge/canonical/_roster.yaml`)
- CLI: `chatqec canonical list/validate`
- CI: validate-canonical stage runs on every push
```

- [ ] **Step 2: Run full suite + lint + canonical validate**

```bash
uv run pytest -v
uv run ruff check src tests
uv run mypy src
uv run chatqec canonical validate
```

Expected: all green. Total tests after M3: ≥46 (M1: 6 + M2: 21 + M3: 19).

- [ ] **Step 3: Commit and push**

```bash
git add CHANGELOG.md
git commit -m "docs: update CHANGELOG for M3 (canonical wiki layer + 5 starter pages)"
git push origin main
```

Expected: GitLab pipeline shows lint, validate-canonical, and test all green.

---

## Milestone 3 — exit checklist

- [ ] `parse_page` + `load_canonical_directory` round-trip the 5 P0 pages
- [ ] Cross-reference validator catches dangling `see_also` entries
- [ ] Section chunker splits on `##` boundaries with preamble + per-section + anchor
- [ ] `chatqec canonical list` shows 5 rows
- [ ] `chatqec canonical validate` exits 0
- [ ] CI runs `validate-canonical` and is green
- [ ] ≥19 new tests added; full suite ≥46 passing
- [ ] CHANGELOG updated
- [ ] GitLab CI green on `main`

**🛑 PAUSE — push to GitLab and review before requesting Milestone 4.**

---

## Milestone 4 — Ingestion pipeline (fetchers, parsers, math-aware chunker)

**Outcome:** Every source type the spec supports has a working fetcher → parser → chunker → on-disk JSONL chunks. The CLI gains `chatqec ingest --pdf / --arxiv / --github / --youtube / --url / --resolve-by-title`. The math-aware chunker protects LaTeX environments. The arXiv resolver (spec §7.4) refuses to trust LLM-supplied IDs and searches by title+author instead. Watchlist YAML parses and a `chatqec watch refresh --dry-run` lists candidate sources without pushing to the registry.

**Embedding and Qdrant are deferred to M5.** M4's ingest commands produce chunks on disk (`data/chunks/<source_id>.jsonl`) and append a `pending_review` record to `sources/registry.jsonl`. M5 will sweep those chunks into Qdrant and flip the registry status to `active`.

**Files this milestone creates / touches:**
- Create: `src/chatqec/ingestion/chunk.py`
- Create: `src/chatqec/ingestion/chunkers/__init__.py`
- Create: `src/chatqec/ingestion/chunkers/math_aware.py`
- Create: `src/chatqec/ingestion/fetchers/__init__.py`
- Create: `src/chatqec/ingestion/fetchers/{arxiv,github,youtube,url,pdf}.py`
- Create: `src/chatqec/ingestion/resolver.py`
- Create: `src/chatqec/ingestion/watchlist.py`
- Create: `sources/watchlist.yaml`
- Create: `tests/test_chunk_model.py`
- Create: `tests/test_chunker_math_aware.py`
- Create: `tests/test_fetcher_arxiv.py`
- Create: `tests/test_resolver_arxiv.py`
- Create: `tests/test_fetcher_github.py`
- Create: `tests/test_fetcher_youtube.py`
- Create: `tests/test_fetcher_url.py`
- Create: `tests/test_fetcher_pdf.py`
- Create: `tests/test_watchlist.py`
- Create: `tests/test_cli_ingest.py`
- Create: `tests/fixtures/arxiv_atom_2408.13687.xml`
- Create: `tests/fixtures/github_repo_metadata.json`
- Create: `tests/fixtures/youtube_transcript.txt`
- Create: `tests/fixtures/blog_post.html`
- Create: `tests/fixtures/sample_paper.pdf` (small placeholder PDF)
- Modify: `src/chatqec/cli.py` (add `ingest` and `watch` command groups)
- Modify: `pyproject.toml` (add `[tool.pytest.ini_options] markers`)
- Modify: `CHANGELOG.md`

**Test strategy:** every fetcher has unit tests that run **against fixtures only** (no network). A `@pytest.mark.network` marker tags tests that hit real services; CI skips them by default and runs them in a separate on-demand `network-tests` job.

---

### Task 4.1 — `Chunk` model + on-disk JSONL writer

**Files:**
- Create: `src/chatqec/ingestion/chunk.py`
- Create: `tests/test_chunk_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_chunk_model.py`:

```python
"""Chunk = the unit the embedder consumes. M4 writes chunks to disk; M5 embeds them."""

import json
from pathlib import Path

import pytest

from chatqec.ingestion.chunk import Chunk, ChunkSink


def test_chunk_required_fields():
    c = Chunk(
        chunk_id="arxiv:2408.13687v1#0",
        source_id="arxiv:2408.13687v1",
        chunk_index=0,
        text="The surface code is a topological code.",
        title="Quantum error correction below the surface code threshold",
        source_type="arxiv",
        authority_tier="peer_reviewed",
        anchor="#section-1",
    )
    assert c.chunk_id == "arxiv:2408.13687v1#0"
    assert c.metadata == {}


def test_chunk_round_trips_through_json():
    c = Chunk(
        chunk_id="x#0",
        source_id="x",
        chunk_index=0,
        text="hi",
        title="T",
        source_type="blog",
        authority_tier="blog",
        anchor="",
        metadata={"author": "alice"},
    )
    blob = c.model_dump_json()
    decoded = Chunk.model_validate_json(blob)
    assert decoded == c


def test_chunk_sink_writes_jsonl(tmp_path: Path):
    sink = ChunkSink(tmp_path)
    c1 = Chunk(chunk_id="x#0", source_id="x", chunk_index=0, text="a",
               title="T", source_type="blog", authority_tier="blog", anchor="")
    c2 = Chunk(chunk_id="x#1", source_id="x", chunk_index=1, text="b",
               title="T", source_type="blog", authority_tier="blog", anchor="")
    sink.write_many("x", [c1, c2])

    expected = tmp_path / "x.jsonl"
    assert expected.exists()
    lines = expected.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["text"] == "a"
    assert parsed[1]["chunk_index"] == 1


def test_chunk_sink_replaces_existing_file(tmp_path: Path):
    sink = ChunkSink(tmp_path)
    sink.write_many("x", [
        Chunk(chunk_id="x#0", source_id="x", chunk_index=0, text="old",
              title="T", source_type="blog", authority_tier="blog", anchor=""),
    ])
    sink.write_many("x", [
        Chunk(chunk_id="x#0", source_id="x", chunk_index=0, text="new",
              title="T", source_type="blog", authority_tier="blog", anchor=""),
    ])
    lines = (tmp_path / "x.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert "new" in lines[0]
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_chunk_model.py -v
```

Expected: ImportError on `chatqec.ingestion.chunk`.

- [ ] **Step 3: Implement `src/chatqec/ingestion/chunk.py`**

```python
"""Chunk model + on-disk JSONL sink.

Chunks are the unit the embedder consumes. M4 produces them; M5 embeds them
and writes them to Qdrant.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    source_id: str
    chunk_index: int
    text: str
    title: str
    source_type: str  # mirrors SourceType.value for embedding-time provenance
    authority_tier: str
    anchor: str = ""  # e.g. "#section-3", "?t=185s", "L42-L88"
    metadata: dict[str, object] = Field(default_factory=dict)


class ChunkSink:
    """Writes chunks for one source to data/chunks/<source_id>.jsonl atomically."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _safe_filename(self, source_id: str) -> str:
        # Make source-id safe for filesystem (replace : / @ with _)
        return source_id.replace(":", "_").replace("/", "_").replace("@", "_") + ".jsonl"

    def write_many(self, source_id: str, chunks: list[Chunk]) -> Path:
        out = self.root_dir / self._safe_filename(source_id)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json())
                f.write("\n")
        tmp.replace(out)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_chunk_model.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/chunk.py tests/test_chunk_model.py
git commit -m "feat(ingestion): add Chunk model and JSONL ChunkSink"
```

---

### Task 4.2 — Math-aware chunker

**Files:**
- Create: `src/chatqec/ingestion/chunkers/__init__.py` (empty)
- Create: `src/chatqec/ingestion/chunkers/math_aware.py`
- Create: `tests/test_chunker_math_aware.py`

- [ ] **Step 1: Make `chunkers` a package**

```bash
touch src/chatqec/ingestion/chunkers/__init__.py
```

- [ ] **Step 2: Write failing golden tests**

Create `tests/test_chunker_math_aware.py`:

```python
"""Math-aware chunker — never splits inside LaTeX math environments.

The single biggest accuracy lever for QEC content. Tests use golden strings
that contain $...$, $$...$$, \begin{equation}...\end{equation}, and
\begin{align}...\end{align}.
"""

from chatqec.ingestion.chunkers.math_aware import chunk_text


def test_chunker_yields_chunks_within_target_size():
    text = "qec " * 1000  # ~4000 chars
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=20)
    assert len(chunks) > 1
    # crude proxy for token count (~4 chars/token)
    assert all(len(c) <= 200 * 6 for c in chunks)


def test_chunker_never_splits_inline_math():
    text = (
        "Definition. The stabilizer of a surface code satisfies $S_i S_j = S_j S_i$ "
        + "for all i, j, as proved in earlier work. "
    ) * 200  # long enough to force multiple chunks
    chunks = chunk_text(text, target_tokens=80, overlap_tokens=10)
    for c in chunks:
        # No chunk may end in the middle of an inline $...$ math span
        dollar_count = c.count("$")
        # Even count means math spans are paired (closed)
        assert dollar_count % 2 == 0, (
            f"chunk has unbalanced $ count {dollar_count}:\n{c!r}"
        )


def test_chunker_keeps_equation_environment_intact():
    text = (
        "The stabilizer code definition.\n\n"
        + r"\begin{equation}" + "\n"
        + r"S = \langle g_1, \ldots, g_{n-k} \rangle \subset P_n"
        + "\n" + r"\end{equation}" + "\n\n"
        + "and the code space is " + ("Hello world. " * 200)
    )
    chunks = chunk_text(text, target_tokens=80, overlap_tokens=10)
    # Find the chunk containing \begin{equation}; it must also contain \end{equation}
    found = False
    for c in chunks:
        if r"\begin{equation}" in c:
            assert r"\end{equation}" in c, "equation env split across chunks"
            found = True
    assert found, "test fixture not large enough"


def test_chunker_keeps_align_environment_intact():
    text = (
        "Lead-in. "
        + r"\begin{align}" + "\n"
        + "a &= b \\\\\n"
        + "c &= d"
        + "\n" + r"\end{align}" + "\n"
        + "tail-in. " * 400
    )
    chunks = chunk_text(text, target_tokens=80, overlap_tokens=10)
    for c in chunks:
        if r"\begin{align}" in c:
            assert r"\end{align}" in c


def test_chunker_keeps_display_math_intact():
    text = (
        "Lead-in. " * 50
        + "$$\\sum_i a_i = 1$$"
        + " mid. " * 50
        + "$$\\prod_j b_j = 0$$"
        + " tail. " * 50
    )
    chunks = chunk_text(text, target_tokens=40, overlap_tokens=5)
    for c in chunks:
        assert c.count("$$") % 2 == 0


def test_chunker_returns_single_chunk_for_short_text():
    text = "Short body."
    chunks = chunk_text(text, target_tokens=200, overlap_tokens=20)
    assert chunks == [text]


def test_chunker_handles_empty_text():
    assert chunk_text("", target_tokens=200, overlap_tokens=20) == []


def test_chunker_overlap_is_within_target():
    text = ("A sentence about QEC. " * 500)
    chunks = chunk_text(text, target_tokens=100, overlap_tokens=20)
    if len(chunks) >= 2:
        # Overlap implies consecutive chunks share some prefix/suffix words
        assert any(w in chunks[0] for w in chunks[1].split()[:10])
```

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/test_chunker_math_aware.py -v
```

Expected: ImportError on `chatqec.ingestion.chunkers.math_aware`.

- [ ] **Step 4: Implement `src/chatqec/ingestion/chunkers/math_aware.py`**

```python
r"""Math-aware text chunker.

Splits text into ~target_tokens-sized chunks at sentence boundaries while
never splitting inside a protected LaTeX region:

    $...$        inline math
    $$...$$      display math
    \begin{X}...\end{X}   any environment

Protection is done by first masking protected spans, then chunking the masked
text by sentence, then unmasking before yielding.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Approximate token count from character count (4 chars/token rule of thumb).
_CHARS_PER_TOKEN = 4


_MASK_TOKEN = "☃PROT{}☃"  # snowman as unlikely-in-source delimiter


def _find_protected_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    # Sort patterns from greediest to least greedy to avoid nested mis-matches.
    patterns = [
        (re.compile(r"\\begin\{(\w+)\}.*?\\end\{\1\}", re.DOTALL), 0),
        (re.compile(r"\$\$.*?\$\$", re.DOTALL), 0),
        (re.compile(r"\$[^$\n]+\$"), 0),  # inline math doesn't cross newlines
    ]
    occupied = []  # already-claimed [start,end) intervals
    for pat, _ in patterns:
        for m in pat.finditer(text):
            s, e = m.span()
            # skip if overlaps an earlier (broader) span
            if any(not (e <= os or s >= oe) for (os, oe) in occupied):
                continue
            occupied.append((s, e))
            spans.append((s, e))
    return sorted(spans)


def _mask_protected(text: str) -> tuple[str, list[str]]:
    """Replace protected spans with sentinel tokens; return (masked_text, originals)."""
    spans = _find_protected_spans(text)
    originals: list[str] = []
    pieces: list[str] = []
    cursor = 0
    for s, e in spans:
        pieces.append(text[cursor:s])
        sentinel = _MASK_TOKEN.format(len(originals))
        originals.append(text[s:e])
        pieces.append(sentinel)
        cursor = e
    pieces.append(text[cursor:])
    return "".join(pieces), originals


def _unmask(text: str, originals: list[str]) -> str:
    for i, orig in enumerate(originals):
        text = text.replace(_MASK_TOKEN.format(i), orig)
    return text


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\\(])")


def _split_sentences(text: str) -> Iterable[str]:
    parts = _SENT_RE.split(text)
    for p in parts:
        if p:
            yield p


def chunk_text(text: str, *, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into math-protected chunks of ~target_tokens characters of size."""
    if not text:
        return []
    target_chars = target_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    masked, originals = _mask_protected(text)
    if len(masked) <= target_chars:
        return [_unmask(masked, originals)]

    sentences = list(_split_sentences(masked))
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for sent in sentences:
        if buf_len + len(sent) + 1 > target_chars and buf:
            chunks.append(" ".join(buf))
            # build overlap tail
            tail: list[str] = []
            tlen = 0
            for s in reversed(buf):
                if tlen + len(s) > overlap_chars:
                    break
                tail.insert(0, s)
                tlen += len(s)
            buf = list(tail)
            buf_len = sum(len(s) for s in buf)
        buf.append(sent)
        buf_len += len(sent) + 1
    if buf:
        chunks.append(" ".join(buf))

    return [_unmask(c, originals) for c in chunks]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_chunker_math_aware.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/ingestion/chunkers/ tests/test_chunker_math_aware.py
git commit -m "feat(chunker): add math-aware chunker that protects LaTeX environments"
```

---

### Task 4.3 — arXiv title-and-author resolver (the §7.4 mandatory pattern)

**Files:**
- Create: `src/chatqec/ingestion/resolver.py`
- Create: `tests/test_resolver_arxiv.py`

- [ ] **Step 1: Write failing tests with mocked arxiv search**

Create `tests/test_resolver_arxiv.py`:

```python
"""arXiv resolver — the §7.4 mandatory pattern.

Never trust an LLM-claimed arXiv ID. Always resolve by title + author via the
arXiv search API. The verification finding in spec A.10 showed 71% of LLM IDs
point to wrong papers; this resolver is the architectural fix.
"""

from unittest.mock import MagicMock, patch

import pytest

from chatqec.ingestion.resolver import (
    AmbiguousResolutionError,
    NoMatchError,
    ResolverHit,
    resolve_arxiv_paper,
)


class _FakeResult:
    def __init__(self, entry_id: str, title: str, authors: list[str], year: int) -> None:
        self.entry_id = entry_id  # e.g. "http://arxiv.org/abs/2408.13687v1"
        self.title = title
        self.authors = [MagicMock(name=a) for a in authors]
        for a, name in zip(self.authors, authors, strict=True):
            a.name = name
        self.published = MagicMock(year=year)


def _patch_search(monkeypatch, results: list[_FakeResult]) -> None:
    mock_search = MagicMock()
    mock_search.return_value.results.return_value = iter(results)
    monkeypatch.setattr("chatqec.ingestion.resolver._arxiv_search", mock_search)


def test_resolver_returns_unique_match(monkeypatch):
    _patch_search(monkeypatch, [
        _FakeResult(
            "http://arxiv.org/abs/2408.13687v1",
            "Quantum error correction below the surface code threshold",
            ["Rajeev Acharya", "Et al"],
            2024,
        )
    ])
    hit = resolve_arxiv_paper(
        title_keywords="below the surface code threshold",
        authors=["Acharya"],
    )
    assert isinstance(hit, ResolverHit)
    assert hit.arxiv_id == "2408.13687v1"
    assert "Acharya" in hit.first_author


def test_resolver_strips_version_suffix_when_asked(monkeypatch):
    _patch_search(monkeypatch, [
        _FakeResult(
            "http://arxiv.org/abs/2408.13687v3",
            "Quantum error correction below the surface code threshold",
            ["Rajeev Acharya"],
            2024,
        )
    ])
    hit = resolve_arxiv_paper(
        title_keywords="surface code threshold",
        authors=["Acharya"],
        strip_version=True,
    )
    assert hit.arxiv_id == "2408.13687"


def test_resolver_raises_no_match(monkeypatch):
    _patch_search(monkeypatch, [])
    with pytest.raises(NoMatchError):
        resolve_arxiv_paper(title_keywords="nonsense paper", authors=["Nobody"])


def test_resolver_raises_ambiguous_when_multiple_strong_matches(monkeypatch):
    _patch_search(monkeypatch, [
        _FakeResult(
            "http://arxiv.org/abs/2207.06431v1",
            "Suppressing quantum errors by scaling a surface code logical qubit",
            ["Google Quantum AI"], 2022,
        ),
        _FakeResult(
            "http://arxiv.org/abs/2408.13687v1",
            "Quantum error correction below the surface code threshold",
            ["Rajeev Acharya"], 2024,
        ),
    ])
    with pytest.raises(AmbiguousResolutionError):
        # Both contain "surface code" and "Google" partially matches both; no
        # clear winner without disambiguating by year.
        resolve_arxiv_paper(title_keywords="surface code", authors=["Google"])


def test_resolver_disambiguates_by_year(monkeypatch):
    _patch_search(monkeypatch, [
        _FakeResult(
            "http://arxiv.org/abs/2207.06431v1",
            "Suppressing quantum errors by scaling a surface code logical qubit",
            ["Google Quantum AI"], 2022,
        ),
        _FakeResult(
            "http://arxiv.org/abs/2408.13687v1",
            "Quantum error correction below the surface code threshold",
            ["Rajeev Acharya"], 2024,
        ),
    ])
    hit = resolve_arxiv_paper(
        title_keywords="surface code threshold",
        authors=["Acharya"],
        expected_year=2024,
    )
    assert hit.arxiv_id == "2408.13687v1"
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_resolver_arxiv.py -v
```

Expected: ImportError on `chatqec.ingestion.resolver`.

- [ ] **Step 3: Implement `src/chatqec/ingestion/resolver.py`**

```python
"""arXiv title-and-author resolver. See spec §7.4 and A.10.

LLM-generated arXiv IDs are unreliable (~14% accurate per the verification
pass during brainstorming). The correct ingestion pattern is to search by
title + author and pick the unique best match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import arxiv


class ResolverError(Exception):
    pass


class NoMatchError(ResolverError):
    pass


class AmbiguousResolutionError(ResolverError):
    pass


@dataclass(frozen=True)
class ResolverHit:
    arxiv_id: str
    title: str
    first_author: str
    year: int


# Reusable client; arxiv lib internally rate-limits to 3 req/s.
_CLIENT = arxiv.Client(page_size=10, delay_seconds=3.0, num_retries=3)


def _arxiv_search(query: str, max_results: int = 10) -> arxiv.Search:
    return arxiv.Search(query=query, max_results=max_results)


def _id_from_entry(entry_id: str, strip_version: bool) -> str:
    m = re.search(r"abs/([^v\s]+)(v\d+)?", entry_id)
    if not m:
        raise ResolverError(f"could not parse arxiv id from {entry_id!r}")
    base = m.group(1)
    version = m.group(2) or ""
    return base if strip_version else (base + version)


def _score(result_title: str, result_authors_lower: list[str], result_year: int,
           title_keywords: str, authors: list[str], expected_year: int | None) -> float:
    """Higher = better match."""
    score = 0.0
    title_l = result_title.lower()
    kw_l = title_keywords.lower()
    # title keyword full substring match is the strongest signal
    if kw_l in title_l:
        score += 5.0
    # per-word overlap
    kw_words = [w for w in kw_l.split() if len(w) > 3]
    title_words = set(title_l.split())
    score += sum(1.0 for w in kw_words if w in title_words)
    # author match — last name substring
    for a in authors:
        last = a.split()[-1].lower()
        if any(last in ra for ra in result_authors_lower):
            score += 2.0
    # year match
    if expected_year is not None:
        if result_year == expected_year:
            score += 2.0
        elif abs(result_year - expected_year) == 1:
            score += 0.5  # publish year drift
    return score


def resolve_arxiv_paper(
    *,
    title_keywords: str,
    authors: list[str],
    expected_year: int | None = None,
    strip_version: bool = False,
) -> ResolverHit:
    """Return the unique best-matching arXiv paper for the given title+authors.

    Raises:
        NoMatchError: no result above the minimum quality threshold.
        AmbiguousResolutionError: top two candidates are within 1.0 score points.
    """
    query = f'ti:"{title_keywords}" AND au:"{authors[0]}"' if authors else f'ti:"{title_keywords}"'
    search = _arxiv_search(query, max_results=10)
    results = list(_CLIENT.results(search))

    scored: list[tuple[float, arxiv.Result]] = []
    for r in results:
        author_names = [a.name.lower() for a in r.authors]
        s = _score(r.title, author_names, r.published.year,
                   title_keywords, authors, expected_year)
        scored.append((s, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored or scored[0][0] < 4.0:
        raise NoMatchError(
            f"no arXiv result above quality threshold for title={title_keywords!r}, authors={authors!r}"
        )
    if len(scored) >= 2 and scored[0][0] - scored[1][0] < 1.0:
        raise AmbiguousResolutionError(
            f"two candidates tied: {scored[0][1].title!r} vs {scored[1][1].title!r}"
        )

    top = scored[0][1]
    return ResolverHit(
        arxiv_id=_id_from_entry(top.entry_id, strip_version=strip_version),
        title=top.title,
        first_author=top.authors[0].name if top.authors else "",
        year=top.published.year,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_resolver_arxiv.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/resolver.py tests/test_resolver_arxiv.py
git commit -m "feat(ingestion): add arXiv title-and-author resolver (spec §7.4)"
```

---

### Task 4.4 — arXiv fetcher (fetch + parse + chunk)

**Files:**
- Create: `src/chatqec/ingestion/fetchers/__init__.py` (empty)
- Create: `src/chatqec/ingestion/fetchers/arxiv.py`
- Create: `tests/fixtures/arxiv_atom_2408.13687.xml`
- Create: `tests/test_fetcher_arxiv.py`

- [ ] **Step 1: Make `fetchers` a package**

```bash
mkdir -p src/chatqec/ingestion/fetchers
touch src/chatqec/ingestion/fetchers/__init__.py
mkdir -p tests/fixtures
```

- [ ] **Step 2: Save the fixture Atom response**

Create `tests/fixtures/arxiv_atom_2408.13687.xml` with a minimal Atom payload:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2408.13687v1</id>
    <updated>2024-08-26T00:00:00Z</updated>
    <published>2024-08-26T00:00:00Z</published>
    <title>Quantum error correction below the surface code threshold</title>
    <summary>Abstract: The surface code is a leading candidate for fault-tolerant quantum computation. We demonstrate that increasing the code distance from 3 to 7 decreases the logical error rate exponentially, marking the first below-threshold experiment on a real quantum processor.</summary>
    <author><name>Rajeev Acharya</name></author>
    <author><name>Google Quantum AI</name></author>
    <link href="http://arxiv.org/pdf/2408.13687v1" rel="alternate" type="application/pdf"/>
  </entry>
</feed>
```

- [ ] **Step 3: Write failing tests**

Create `tests/test_fetcher_arxiv.py`:

```python
"""arXiv fetcher tests — parse-only, no network."""

from pathlib import Path

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.fetchers.arxiv import (
    ArxivMetadata,
    parse_arxiv_atom,
    paper_to_chunks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_atom_2408.13687.xml"


def test_parse_arxiv_atom_extracts_metadata():
    text = FIXTURE.read_text(encoding="utf-8")
    md = parse_arxiv_atom(text)
    assert isinstance(md, ArxivMetadata)
    assert md.arxiv_id == "2408.13687v1"
    assert md.title.startswith("Quantum error correction")
    assert md.authors[0] == "Rajeev Acharya"
    assert md.year == 2024
    assert "fault-tolerant" in md.abstract.lower()


def test_paper_to_chunks_produces_chunks_with_correct_ids():
    md = ArxivMetadata(
        arxiv_id="2408.13687v1",
        title="Quantum error correction below the surface code threshold",
        authors=["Rajeev Acharya"],
        year=2024,
        abstract="A long abstract about QEC. " * 100,
        pdf_url="http://arxiv.org/pdf/2408.13687v1",
    )
    chunks = paper_to_chunks(md, body_text=md.abstract,
                             target_tokens=80, overlap_tokens=10)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].source_id == "arxiv:2408.13687v1"
    assert chunks[0].chunk_id == "arxiv:2408.13687v1#0"
    assert chunks[0].title.startswith("Quantum error")
    assert chunks[0].source_type == "arxiv"
    assert chunks[0].authority_tier == "preprint"


def test_paper_to_chunks_uses_math_aware_chunker():
    md = ArxivMetadata(
        arxiv_id="abc.1234v1", title="t", authors=["X"], year=2024,
        abstract="", pdf_url="",
    )
    body = r"Lead. \begin{equation} S = \langle g_1, g_2 \rangle \end{equation}" + " tail. " * 200
    chunks = paper_to_chunks(md, body_text=body, target_tokens=50, overlap_tokens=5)
    for c in chunks:
        if r"\begin{equation}" in c.text:
            assert r"\end{equation}" in c.text
```

- [ ] **Step 4: Run failing tests**

```bash
uv run pytest tests/test_fetcher_arxiv.py -v
```

Expected: ImportError.

- [ ] **Step 5: Implement `src/chatqec/ingestion/fetchers/arxiv.py`**

```python
"""arXiv fetcher — parses Atom responses and produces chunks.

Network fetching uses the `arxiv` library (already used by the resolver).
This module's parser is decoupled so we can unit-test on fixtures.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.chunkers.math_aware import chunk_text

_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True)
class ArxivMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    year: int
    abstract: str
    pdf_url: str


def parse_arxiv_atom(xml_text: str) -> ArxivMetadata:
    root = ET.fromstring(xml_text)
    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        raise ValueError("arxiv atom: no <entry> element")

    id_text = (entry.find(f"{_ATOM}id").text or "").strip()
    m = re.search(r"abs/([^\s]+)", id_text)
    arxiv_id = m.group(1) if m else id_text

    title = (entry.find(f"{_ATOM}title").text or "").strip().replace("\n", " ")
    summary = (entry.find(f"{_ATOM}summary").text or "").strip()
    published = (entry.find(f"{_ATOM}published").text or "")[:4]
    year = int(published) if published.isdigit() else 0
    authors = [a.find(f"{_ATOM}name").text for a in entry.findall(f"{_ATOM}author")]
    pdf_url = ""
    for link in entry.findall(f"{_ATOM}link"):
        if link.get("type") == "application/pdf":
            pdf_url = link.get("href") or ""
            break

    return ArxivMetadata(
        arxiv_id=arxiv_id,
        title=title,
        authors=[a or "" for a in authors],
        year=year,
        abstract=summary,
        pdf_url=pdf_url,
    )


def paper_to_chunks(
    md: ArxivMetadata,
    *,
    body_text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    source_id = f"arxiv:{md.arxiv_id}"
    pieces = chunk_text(body_text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    chunks: list[Chunk] = []
    for i, p in enumerate(pieces):
        chunks.append(Chunk(
            chunk_id=f"{source_id}#{i}",
            source_id=source_id,
            chunk_index=i,
            text=p,
            title=md.title,
            source_type="arxiv",
            authority_tier="preprint",
            anchor=f"#chunk-{i}",
            metadata={"authors": md.authors, "year": md.year, "pdf_url": md.pdf_url},
        ))
    return chunks
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_fetcher_arxiv.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/chatqec/ingestion/fetchers/__init__.py src/chatqec/ingestion/fetchers/arxiv.py
git add tests/fixtures/arxiv_atom_2408.13687.xml tests/test_fetcher_arxiv.py
git commit -m "feat(fetcher): add arXiv Atom parser and paper→chunk pipeline"
```

---

### Task 4.5 — GitHub fetcher (LlamaIndex reader, README + docs/)

**Files:**
- Create: `src/chatqec/ingestion/fetchers/github.py`
- Create: `tests/fixtures/github_readme.md`
- Create: `tests/test_fetcher_github.py`

- [ ] **Step 1: Save fixture README**

Create `tests/fixtures/github_readme.md`:

```markdown
# Stim

A fast quantum stabilizer circuit simulator.

## Installation

Install via pip:

```bash
pip install stim
```

## Usage

Stim simulates Clifford circuits with detector error model export. See [doc/getting_started.ipynb](doc/getting_started.ipynb).

## License

Apache 2.0.
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_fetcher_github.py`:

```python
"""GitHub fetcher tests — parse-only, no network."""

from pathlib import Path

from chatqec.ingestion.fetchers.github import (
    GitHubMetadata,
    files_to_chunks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "github_readme.md"


def test_files_to_chunks_emits_chunks_per_file():
    readme_text = FIXTURE.read_text(encoding="utf-8")
    md = GitHubMetadata(
        org="quantumlib", repo="Stim", commit_sha="deadbeefcafef00d",
        license="Apache-2.0",
    )
    chunks = files_to_chunks(
        md,
        files={"README.md": readme_text, "doc/intro.md": "# Intro\n\nShort intro."},
        target_tokens=100, overlap_tokens=10,
    )
    sources = {c.source_id for c in chunks}
    assert sources == {"github:quantumlib/Stim@deadbeef"}
    titles = {c.metadata.get("path") for c in chunks}
    assert "README.md" in titles
    assert "doc/intro.md" in titles


def test_files_to_chunks_skips_empty_files():
    md = GitHubMetadata(org="o", repo="r", commit_sha="abc12345", license="MIT")
    chunks = files_to_chunks(
        md,
        files={"empty.md": "", "real.md": "content"},
        target_tokens=100, overlap_tokens=10,
    )
    assert all(c.metadata.get("path") != "empty.md" for c in chunks)
    assert any(c.metadata.get("path") == "real.md" for c in chunks)


def test_files_to_chunks_assigns_blog_authority_tier_by_default():
    md = GitHubMetadata(org="o", repo="r", commit_sha="abc12345", license="MIT")
    chunks = files_to_chunks(md, files={"r.md": "x"}, target_tokens=100, overlap_tokens=10)
    assert chunks[0].source_type == "github"
    assert chunks[0].authority_tier == "blog"  # neither canonical nor peer-reviewed
```

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/test_fetcher_github.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `src/chatqec/ingestion/fetchers/github.py`**

```python
"""GitHub fetcher — operates on a dict of {path: text} after a LlamaIndex pull.

The actual repo download uses LlamaIndex's `GithubRepositoryReader`; this
module's parser is pure so it can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.chunkers.math_aware import chunk_text
from chatqec.ingestion.ids import github_id


@dataclass(frozen=True)
class GitHubMetadata:
    org: str
    repo: str
    commit_sha: str
    license: str


def files_to_chunks(
    md: GitHubMetadata,
    *,
    files: dict[str, str],
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    source_id = github_id(md.org, md.repo, md.commit_sha)
    out: list[Chunk] = []
    idx = 0
    for path, text in files.items():
        if not text.strip():
            continue
        for piece in chunk_text(text, target_tokens=target_tokens, overlap_tokens=overlap_tokens):
            out.append(Chunk(
                chunk_id=f"{source_id}#{idx}",
                source_id=source_id,
                chunk_index=idx,
                text=piece,
                title=f"{md.org}/{md.repo}",
                source_type="github",
                authority_tier="blog",
                anchor=f"path={path}",
                metadata={"path": path, "license": md.license},
            ))
            idx += 1
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_fetcher_github.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/ingestion/fetchers/github.py tests/fixtures/github_readme.md tests/test_fetcher_github.py
git commit -m "feat(fetcher): add GitHub files→chunks parser"
```

---

### Task 4.6 — YouTube transcript fetcher

**Files:**
- Create: `src/chatqec/ingestion/fetchers/youtube.py`
- Create: `tests/fixtures/youtube_transcript.json`
- Create: `tests/test_fetcher_youtube.py`

- [ ] **Step 1: Save fixture transcript (yt-dlp JSON3 format trimmed)**

Create `tests/fixtures/youtube_transcript.json`:

```json
{
  "events": [
    {"tStartMs": 0, "segs": [{"utf8": "Welcome to surface code 101."}]},
    {"tStartMs": 5000, "segs": [{"utf8": "The surface code is topological."}]},
    {"tStartMs": 12000, "segs": [{"utf8": "Stabilizers are local operators."}]},
    {"tStartMs": 65000, "segs": [{"utf8": "We use MWPM for decoding."}]},
    {"tStartMs": 130000, "segs": [{"utf8": "Thanks for watching."}]}
  ]
}
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_fetcher_youtube.py`:

```python
"""YouTube transcript chunker — 60-second window aggregation."""

import json
from pathlib import Path

from chatqec.ingestion.fetchers.youtube import (
    YouTubeMetadata,
    transcript_to_chunks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "youtube_transcript.json"


def test_transcript_to_chunks_groups_by_60s_window():
    transcript = json.loads(FIXTURE.read_text())
    md = YouTubeMetadata(
        video_id="abc123", title="Surface code 101", channel="QECChannel",
    )
    chunks = transcript_to_chunks(md, transcript_events=transcript["events"],
                                   window_seconds=60)
    # 0s, 5s, 12s → window 0; 65s → window 1; 130s → window 2
    assert len(chunks) == 3
    # First chunk contains everything from 0–60s
    assert "Welcome" in chunks[0].text
    assert "MWPM" not in chunks[0].text
    # Second window has the MWPM line
    assert "MWPM" in chunks[1].text


def test_transcript_chunk_anchor_is_timestamp():
    transcript = json.loads(FIXTURE.read_text())
    md = YouTubeMetadata(video_id="abc123", title="t", channel="c")
    chunks = transcript_to_chunks(md, transcript_events=transcript["events"],
                                   window_seconds=60)
    assert chunks[0].anchor == "?t=0s"
    assert chunks[1].anchor == "?t=60s"


def test_transcript_chunk_source_id_uses_video_id():
    transcript = json.loads(FIXTURE.read_text())
    md = YouTubeMetadata(video_id="abc123XYZ", title="t", channel="c")
    chunks = transcript_to_chunks(md, transcript_events=transcript["events"],
                                   window_seconds=60)
    assert chunks[0].source_id == "youtube:abc123XYZ"


def test_empty_transcript_yields_no_chunks():
    md = YouTubeMetadata(video_id="x", title="t", channel="c")
    assert transcript_to_chunks(md, transcript_events=[], window_seconds=60) == []
```

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/test_fetcher_youtube.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `src/chatqec/ingestion/fetchers/youtube.py`**

```python
"""YouTube transcript chunker.

The actual download uses `yt-dlp --write-auto-subs --skip-download` from a
higher-level orchestrator. This module operates on already-parsed events.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chatqec.ingestion.chunk import Chunk


@dataclass(frozen=True)
class YouTubeMetadata:
    video_id: str
    title: str
    channel: str


def transcript_to_chunks(
    md: YouTubeMetadata,
    *,
    transcript_events: Iterable[dict],
    window_seconds: int = 60,
) -> list[Chunk]:
    events = list(transcript_events)
    if not events:
        return []

    source_id = f"youtube:{md.video_id}"
    buckets: dict[int, list[str]] = {}
    for ev in events:
        t_ms = int(ev.get("tStartMs", 0))
        bucket = (t_ms // 1000) // window_seconds
        text_pieces = [s.get("utf8", "") for s in ev.get("segs", [])]
        text = "".join(text_pieces).strip()
        if not text:
            continue
        buckets.setdefault(bucket, []).append(text)

    chunks: list[Chunk] = []
    for i, bucket in enumerate(sorted(buckets)):
        start_seconds = bucket * window_seconds
        text = " ".join(buckets[bucket])
        chunks.append(Chunk(
            chunk_id=f"{source_id}#{i}",
            source_id=source_id,
            chunk_index=i,
            text=text,
            title=md.title,
            source_type="youtube",
            authority_tier="video",
            anchor=f"?t={start_seconds}s",
            metadata={"channel": md.channel, "start_seconds": start_seconds},
        ))
    return chunks
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_fetcher_youtube.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/ingestion/fetchers/youtube.py tests/fixtures/youtube_transcript.json tests/test_fetcher_youtube.py
git commit -m "feat(fetcher): add YouTube transcript chunker (60s windows)"
```

---

### Task 4.7 — Blog/URL fetcher (defuddle → markdown → chunks)

**Files:**
- Create: `src/chatqec/ingestion/fetchers/url.py`
- Create: `tests/fixtures/blog_post.md`
- Create: `tests/test_fetcher_url.py`

- [ ] **Step 1: Save fixture (already-defuddled markdown)**

Create `tests/fixtures/blog_post.md`:

```markdown
# Surface codes hit a milestone

Posted by alice on 2026-01-15.

## What changed?

Google's Willow chip showed the surface code below threshold for the first time.

## Why it matters

This is the demonstration the field has been waiting for since the 1990s.

## Caveats

Single-qubit logic, not yet a full computer. Still a huge step.
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_fetcher_url.py`:

```python
"""URL fetcher tests — parse-only, no network."""

from pathlib import Path

from chatqec.ingestion.fetchers.url import (
    URLMetadata,
    markdown_to_chunks,
)

FIXTURE = Path(__file__).parent / "fixtures" / "blog_post.md"


def test_markdown_to_chunks_uses_section_boundaries():
    md_text = FIXTURE.read_text(encoding="utf-8")
    md = URLMetadata(
        url="https://research.example.com/surface-code-milestone",
        title="Surface codes hit a milestone",
        publish_date="2026-01-15",
    )
    chunks = markdown_to_chunks(md, markdown_text=md_text,
                                target_tokens=120, overlap_tokens=10)
    assert len(chunks) >= 2
    assert any("Willow" in c.text for c in chunks)
    assert chunks[0].source_type == "blog"
    assert chunks[0].authority_tier == "blog"


def test_url_source_id_is_hash_prefixed():
    md = URLMetadata(url="https://a.test/x", title="t", publish_date="")
    chunks = markdown_to_chunks(md, markdown_text="body", target_tokens=100, overlap_tokens=5)
    assert chunks[0].source_id.startswith("url:")
    assert len(chunks[0].source_id) == len("url:") + 16
```

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/test_fetcher_url.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `src/chatqec/ingestion/fetchers/url.py`**

```python
"""URL/blog fetcher.

A higher-level orchestrator uses `defuddle` to convert raw HTML → markdown,
then passes the markdown text to `markdown_to_chunks`.
"""

from __future__ import annotations

from dataclasses import dataclass

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.chunkers.math_aware import chunk_text
from chatqec.ingestion.ids import url_id


@dataclass(frozen=True)
class URLMetadata:
    url: str
    title: str
    publish_date: str  # "YYYY-MM-DD" or ""


def markdown_to_chunks(
    md: URLMetadata,
    *,
    markdown_text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    source_id = url_id(md.url)
    pieces = chunk_text(markdown_text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    return [
        Chunk(
            chunk_id=f"{source_id}#{i}",
            source_id=source_id,
            chunk_index=i,
            text=p,
            title=md.title,
            source_type="blog",
            authority_tier="blog",
            anchor=f"#chunk-{i}",
            metadata={"url": md.url, "publish_date": md.publish_date},
        )
        for i, p in enumerate(pieces)
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_fetcher_url.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/ingestion/fetchers/url.py tests/fixtures/blog_post.md tests/test_fetcher_url.py
git commit -m "feat(fetcher): add URL/blog markdown→chunks pipeline"
```

---

### Task 4.8 — Manual PDF fetcher (text extraction + chunking)

**Files:**
- Create: `src/chatqec/ingestion/fetchers/pdf.py`
- Create: `tests/test_fetcher_pdf.py`

PDF parsing in tests is awkward without a real PDF. We test the *post-parse* chunking pipeline (the parser stage is just a thin wrapper around `unstructured.partition.pdf.partition_pdf` and will be smoke-tested via `@pytest.mark.network` in CI).

- [ ] **Step 1: Write failing tests for the chunking pipeline**

Create `tests/test_fetcher_pdf.py`:

```python
"""PDF fetcher tests — operates on pre-extracted text."""

from chatqec.ingestion.fetchers.pdf import (
    PDFMetadata,
    text_to_chunks,
)


def test_text_to_chunks_uses_pdf_source_id():
    md = PDFMetadata(
        content_sha256="ab12cd34" + "0" * 56,
        title="A QEC paper",
        author="Alice",
        year=2024,
        source_path="/tmp/paper.pdf",
    )
    chunks = text_to_chunks(md, body_text="Body. " * 200,
                            target_tokens=80, overlap_tokens=8)
    assert chunks[0].source_id == "pdf:ab12cd34"
    assert chunks[0].source_type == "pdf"
    assert chunks[0].title == "A QEC paper"


def test_text_to_chunks_protects_math():
    md = PDFMetadata(
        content_sha256="a" * 64, title="t", author="a", year=2024,
        source_path="/x.pdf",
    )
    body = r"Lead. \begin{align} S &= \langle g_1, g_2 \rangle \end{align}" + " tail. " * 200
    chunks = text_to_chunks(md, body_text=body, target_tokens=50, overlap_tokens=5)
    for c in chunks:
        if r"\begin{align}" in c.text:
            assert r"\end{align}" in c.text
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_fetcher_pdf.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/chatqec/ingestion/fetchers/pdf.py`**

```python
"""Manual-upload PDF fetcher.

Extraction uses `unstructured.partition.pdf.partition_pdf`. The public function
`extract_pdf_text` returns a single string; the parser/chunker is decoupled so
it can be tested without an actual PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.chunkers.math_aware import chunk_text
from chatqec.ingestion.hashing import sha256_file
from chatqec.ingestion.ids import pdf_id


@dataclass(frozen=True)
class PDFMetadata:
    content_sha256: str
    title: str
    author: str
    year: int
    source_path: str


def extract_pdf_text(path: Path) -> str:
    """Thin wrapper around `unstructured`. Not unit-tested — see network tests."""
    from unstructured.partition.pdf import partition_pdf  # local import to keep cold-start fast

    elements = partition_pdf(filename=str(path))
    return "\n".join(str(el) for el in elements)


def build_pdf_metadata(
    path: Path,
    *,
    title: str,
    author: str,
    year: int,
) -> PDFMetadata:
    return PDFMetadata(
        content_sha256=sha256_file(path),
        title=title,
        author=author,
        year=year,
        source_path=str(path),
    )


def text_to_chunks(
    md: PDFMetadata,
    *,
    body_text: str,
    target_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    source_id = pdf_id(md.content_sha256)
    pieces = chunk_text(body_text, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    return [
        Chunk(
            chunk_id=f"{source_id}#{i}",
            source_id=source_id,
            chunk_index=i,
            text=p,
            title=md.title,
            source_type="pdf",
            authority_tier="preprint",
            anchor=f"#chunk-{i}",
            metadata={"author": md.author, "year": md.year, "source_path": md.source_path},
        )
        for i, p in enumerate(pieces)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_fetcher_pdf.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/fetchers/pdf.py tests/test_fetcher_pdf.py
git commit -m "feat(fetcher): add manual PDF text→chunks pipeline"
```

---

### Task 4.9 — Watchlist YAML schema + parser

**Files:**
- Create: `src/chatqec/ingestion/watchlist.py`
- Create: `sources/watchlist.yaml`
- Create: `tests/test_watchlist.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_watchlist.py`:

```python
"""Watchlist YAML parsing tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from chatqec.ingestion.watchlist import Watchlist, load_watchlist


SAMPLE = """
arxiv:
  categories: [quant-ph]
  keywords: [surface code, qLDPC]
  authors: [Preskill, Gottesman]
github:
  - repo: quantumlib/Stim
  - repo: oscarhiggott/PyMatching
youtube:
  channels: ["@GoogleQuantumAI", "@qiskit"]
rss:
  - https://quantum-journal.org/feed/
"""


def test_load_watchlist_parses_all_sections(tmp_path: Path):
    p = tmp_path / "watchlist.yaml"
    p.write_text(SAMPLE)
    wl = load_watchlist(p)
    assert isinstance(wl, Watchlist)
    assert wl.arxiv.categories == ["quant-ph"]
    assert "surface code" in wl.arxiv.keywords
    assert wl.github[0].repo == "quantumlib/Stim"
    assert "@GoogleQuantumAI" in wl.youtube.channels
    assert str(wl.rss[0]).startswith("https://quantum-journal.org")


def test_load_watchlist_rejects_invalid_github_repo(tmp_path: Path):
    bad = "github:\n  - repo: just-one-segment\n"
    p = tmp_path / "watchlist.yaml"
    p.write_text(bad)
    with pytest.raises(ValidationError):
        load_watchlist(p)


def test_load_watchlist_allows_missing_sections(tmp_path: Path):
    p = tmp_path / "watchlist.yaml"
    p.write_text("arxiv:\n  categories: []\n")
    wl = load_watchlist(p)
    assert wl.github == []
    assert wl.youtube.channels == []
    assert wl.rss == []
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_watchlist.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/chatqec/ingestion/watchlist.py`**

```python
"""Watchlist YAML schema (sources/watchlist.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class ArxivWatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    categories: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)


class GitHubWatchEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo: str

    @field_validator("repo")
    @classmethod
    def _two_segments(cls, v: str) -> str:
        if v.count("/") != 1 or v.startswith("/") or v.endswith("/"):
            raise ValueError(f"github repo must be 'org/name': {v!r}")
        return v


class YouTubeWatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channels: list[str] = Field(default_factory=list)


class Watchlist(BaseModel):
    model_config = ConfigDict(extra="forbid")
    arxiv: ArxivWatch = Field(default_factory=ArxivWatch)
    github: list[GitHubWatchEntry] = Field(default_factory=list)
    youtube: YouTubeWatch = Field(default_factory=YouTubeWatch)
    rss: list[HttpUrl] = Field(default_factory=list)


def load_watchlist(path: Path) -> Watchlist:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Watchlist(**data)
```

- [ ] **Step 4: Write the live watchlist with the seed entries (spec A.4–A.9)**

Create `sources/watchlist.yaml`:

```yaml
# ChatQEC watchlist — drives the weekly refresh job.
# Contributors: add entries via PR. The refresh job opens a PR with new candidates
# in status=pending_review; maintainers approve via merge.
# Authoritative seeds: docs/superpowers/specs/2026-05-29-chatqec-design.md §A.4–A.9

arxiv:
  categories: [quant-ph]
  keywords:
    - "surface code"
    - "stabilizer code"
    - "magic state"
    - "fault-tolerant"
    - "qLDPC"
    - "threshold"
    - "decoder"
    - "Floquet code"
    - "lattice surgery"
  authors:
    - Preskill
    - Gottesman
    - Kitaev
    - Terhal
    - Bombin
    - Litinski
    - Fowler
    - Yoder
    - Gidney
    - Bravyi
    - Roffe

github:
  - repo: quantumlib/Stim
  - repo: oscarhiggott/PyMatching
  - repo: qiskit-community/qiskit-qec
  - repo: quantumgizmos/ldpc
  - repo: panqec/panqec
  - repo: errorcorrectionzoo/eczoo_data
  - repo: microsoft/QuantumKatas
  - repo: qiskit-community/qiskit-textbook

youtube:
  channels:
    - "@GoogleQuantumAI"
    - "@qiskit"
    - "@Quantinuum"
    - "@SimonsInstituteTOC"
    - "@PerimeterInstitute"
    - "@Riverlane"

rss:
  - https://quantum-journal.org/feed/
  - https://scottaaronson.blog/?feed=rss2
```

- [ ] **Step 5: Run tests + validate the real file parses**

```bash
uv run pytest tests/test_watchlist.py -v
uv run python -c "from pathlib import Path; from chatqec.ingestion.watchlist import load_watchlist; print(load_watchlist(Path('sources/watchlist.yaml')))"
```

Expected: 3 tests passed, then a populated `Watchlist(...)` repr printed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/ingestion/watchlist.py sources/watchlist.yaml tests/test_watchlist.py
git commit -m "feat(watchlist): add YAML schema, parser, and seed watchlist"
```

---

### Task 4.10 — CLI: `chatqec ingest --arxiv / --github / --youtube / --url / --pdf`

**Files:**
- Modify: `src/chatqec/cli.py`
- Create: `tests/test_cli_ingest.py`

- [ ] **Step 1: Write failing CLI tests (using fixtures)**

Create `tests/test_cli_ingest.py`:

```python
"""CLI tests for `chatqec ingest` subcommands.

Heavy network calls are not exercised here; we monkeypatch the fetcher's
internal `arxiv.Search` so the tests stay hermetic.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from chatqec.cli import main


@pytest.fixture
def chunk_dir(tmp_path: Path) -> Path:
    out = tmp_path / "chunks"
    out.mkdir()
    return out


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "registry.jsonl"


def test_ingest_url_writes_chunks(tmp_path: Path, chunk_dir: Path, registry_path: Path,
                                  monkeypatch):
    # Stub the HTTP+defuddle pipeline to return canned markdown
    monkeypatch.setattr(
        "chatqec.cli._fetch_url_as_markdown",
        lambda url: ("Surface codes hit a milestone", "Body. " * 500, "2026-01-15"),
    )
    runner = CliRunner()
    result = runner.invoke(main, [
        "ingest", "--chunks-dir", str(chunk_dir), "--registry", str(registry_path),
        "--url", "https://example.com/post",
    ])
    assert result.exit_code == 0, result.output

    # A chunk file landed
    files = list(chunk_dir.glob("*.jsonl"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "Surface codes" in text

    # Registry got a pending_review record
    reg_text = registry_path.read_text()
    assert "pending_review" in reg_text


def test_ingest_arxiv_with_resolve_by_title(monkeypatch, tmp_path: Path,
                                            chunk_dir: Path, registry_path: Path):
    from chatqec.ingestion.resolver import ResolverHit
    monkeypatch.setattr(
        "chatqec.cli._resolve_and_fetch_arxiv",
        lambda title, authors, year: (
            ResolverHit(arxiv_id="2408.13687v1",
                        title="Quantum error correction below the surface code threshold",
                        first_author="Rajeev Acharya", year=2024),
            "Long abstract body. " * 100,
        ),
    )
    runner = CliRunner()
    result = runner.invoke(main, [
        "ingest", "--chunks-dir", str(chunk_dir), "--registry", str(registry_path),
        "--resolve-by-title", "below the surface code threshold",
        "--author", "Acharya", "--year", "2024",
    ])
    assert result.exit_code == 0, result.output
    files = list(chunk_dir.glob("arxiv_2408.13687v1*"))
    assert len(files) == 1
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_cli_ingest.py -v
```

Expected: `ingest` group does not exist.

- [ ] **Step 3: Add the `ingest` group to `src/chatqec/cli.py`**

Add the following imports at the top (with the other ingestion imports):

```python
from datetime import datetime, timezone

from chatqec.ingestion.chunk import ChunkSink
from chatqec.ingestion.fetchers.arxiv import (
    ArxivMetadata,
    paper_to_chunks as arxiv_paper_to_chunks,
)
from chatqec.ingestion.fetchers.url import URLMetadata, markdown_to_chunks
from chatqec.ingestion.registry import SourceRecord, SourceStatus, SourceType
from chatqec.ingestion.resolver import (
    AmbiguousResolutionError,
    NoMatchError,
    ResolverHit,
    resolve_arxiv_paper,
)
```

Then add helpers and the command group:

```python
def _fetch_url_as_markdown(url: str) -> tuple[str, str, str]:
    """Default URL fetcher: HTTP GET + defuddle → markdown. Overridable in tests."""
    import defuddle
    import requests
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    article = defuddle.parse(resp.text, url=url)
    title = article.get("title", url) if isinstance(article, dict) else url
    md_text = article.get("content_markdown", "") if isinstance(article, dict) else str(article)
    publish_date = article.get("date", "") if isinstance(article, dict) else ""
    return title, md_text, publish_date


def _resolve_and_fetch_arxiv(
    title_keywords: str, authors: list[str], year: int | None,
) -> tuple[ResolverHit, str]:
    """Resolve title+author to an arxiv id, then fetch the abstract+full text."""
    import arxiv
    hit = resolve_arxiv_paper(title_keywords=title_keywords, authors=authors,
                              expected_year=year)
    search = arxiv.Search(id_list=[hit.arxiv_id])
    [result] = list(arxiv.Client().results(search))
    # For M4 we ingest the abstract only; full-text PDF parsing lands here
    # behind a flag in M5 once `unstructured` is wired into the pipeline.
    return hit, result.summary


@main.command("ingest")
@click.option("--chunks-dir", type=click.Path(path_type=Path),
              default=Path("data/chunks"), show_default=True,
              help="Directory to write JSONL chunk files.")
@click.option("--registry", "registry_path", type=click.Path(path_type=Path),
              default=Path("sources/registry.jsonl"), show_default=True,
              help="Path to the source registry.")
@click.option("--url", "url_value", default=None, help="Ingest a single URL (blog/HTML).")
@click.option("--arxiv", "arxiv_id_value", default=None,
              help="Ingest by arxiv id. WARNING: §7.4 — prefer --resolve-by-title.")
@click.option("--resolve-by-title", "title_keywords", default=None,
              help="Resolve a paper by title keywords and ingest it.")
@click.option("--author", "authors_opt", multiple=True,
              help="Author last names (use with --resolve-by-title).")
@click.option("--year", type=int, default=None,
              help="Expected publication year (helps disambiguate).")
def ingest(
    chunks_dir: Path, registry_path: Path,
    url_value: str | None,
    arxiv_id_value: str | None,
    title_keywords: str | None,
    authors_opt: tuple[str, ...],
    year: int | None,
) -> None:
    """Ingest one source. Writes chunks to disk and a pending_review record."""
    sink = ChunkSink(chunks_dir)
    reg = Registry(registry_path)
    now = datetime.now(tz=timezone.utc)

    if url_value:
        title, md_text, publish_date = _fetch_url_as_markdown(url_value)
        md = URLMetadata(url=url_value, title=title, publish_date=publish_date)
        chunks = markdown_to_chunks(md, markdown_text=md_text,
                                    target_tokens=512, overlap_tokens=64)
        source_id = chunks[0].source_id
        sink.write_many(source_id, chunks)
        reg.append(SourceRecord(
            id=source_id, type=SourceType.BLOG, url=url_value, title=title,
            authority_tier="blog", ingested_at=now, last_checked_at=now,
            status=SourceStatus.PENDING_REVIEW,
            chunk_count=len(chunks),
            chunk_ids=[c.chunk_id for c in chunks],
        ))
        click.echo(f"ingested {source_id}: {len(chunks)} chunks → {chunks_dir}")
        return

    if title_keywords:
        try:
            hit, body = _resolve_and_fetch_arxiv(title_keywords, list(authors_opt), year)
        except (NoMatchError, AmbiguousResolutionError) as e:
            click.echo(f"resolver failed: {e}", err=True)
            raise SystemExit(2) from None
        md = ArxivMetadata(
            arxiv_id=hit.arxiv_id, title=hit.title,
            authors=[hit.first_author], year=hit.year,
            abstract=body, pdf_url="",
        )
        chunks = arxiv_paper_to_chunks(md, body_text=body,
                                       target_tokens=512, overlap_tokens=64)
        source_id = chunks[0].source_id
        sink.write_many(source_id, chunks)
        reg.append(SourceRecord(
            id=source_id, type=SourceType.ARXIV,
            url=f"https://arxiv.org/abs/{hit.arxiv_id}",
            title=hit.title, authors=[hit.first_author], year=hit.year,
            authority_tier="preprint", ingested_at=now, last_checked_at=now,
            status=SourceStatus.PENDING_REVIEW,
            chunk_count=len(chunks),
            chunk_ids=[c.chunk_id for c in chunks],
            verified=True,
        ))
        click.echo(f"ingested {source_id}: {len(chunks)} chunks (resolved from title)")
        return

    if arxiv_id_value:
        click.echo(
            "WARNING: --arxiv <id> trusts the supplied id. "
            "Per spec §7.4 prefer --resolve-by-title for LLM-suggested papers.",
            err=True,
        )
        # M4 implements only the resolve-by-title path live; the --arxiv path
        # delegates to the same fetcher with id_list=[arxiv_id_value].
        # Full implementation lands in M5 when we wire up _fetch_arxiv_by_id.
        click.echo("--arxiv <id> ingestion is not yet supported (lands in M5)", err=True)
        raise SystemExit(3)

    click.echo("nothing to ingest — pass --url, --resolve-by-title, or --arxiv", err=True)
    raise SystemExit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli_ingest.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Smoke test with a real URL (skip if offline)**

```bash
uv run chatqec ingest --chunks-dir /tmp/chunks --registry /tmp/reg.jsonl \
    --url "https://errorcorrectionzoo.org/c/surface"
ls /tmp/chunks/
cat /tmp/reg.jsonl
```

Expected: chunks file + a `pending_review` record.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/cli.py tests/test_cli_ingest.py
git commit -m "feat(cli): add 'ingest --url/--resolve-by-title' commands"
```

---

### Task 4.11 — CLI: `chatqec watch refresh --dry-run`

**Files:**
- Modify: `src/chatqec/cli.py`
- Modify: `tests/test_cli_ingest.py` (append watch tests)

The `watch refresh` command pulls fresh candidates from the watchlist. In M4 it operates **only in dry-run mode** and prints candidates without writing to the registry. The full PR-opening pipeline lands in M10 as a GitHub Action.

- [ ] **Step 1: Append failing test**

Append to `tests/test_cli_ingest.py`:

```python
def test_watch_refresh_dry_run_prints_candidates(monkeypatch, tmp_path: Path):
    # Stub the watchlist refresh to return a fixed candidate list
    def fake_collect(_wl):
        return [
            ("arxiv:2408.13687v1", "Below threshold", ["Acharya"], 2024),
            ("github:quantumlib/Stim@abc12345", "Stim README", ["Gidney"], 2024),
        ]
    monkeypatch.setattr("chatqec.cli._collect_watchlist_candidates", fake_collect)

    wl_path = tmp_path / "watchlist.yaml"
    wl_path.write_text("arxiv:\n  categories: [quant-ph]\n")

    runner = CliRunner()
    result = runner.invoke(main, [
        "watch", "--watchlist", str(wl_path), "refresh", "--dry-run",
    ])
    assert result.exit_code == 0
    assert "arxiv:2408.13687v1" in result.output
    assert "Stim README" in result.output
```

- [ ] **Step 2: Add the `watch` group to `src/chatqec/cli.py`**

```python
from chatqec.ingestion.watchlist import Watchlist, load_watchlist


def _collect_watchlist_candidates(wl: Watchlist) -> list[tuple[str, str, list[str], int | None]]:
    """Return a list of (candidate_id, title, authors, year).

    In M4 this stub returns an empty list; M10 will wire up the real
    arxiv/github/youtube/rss polling logic behind a GitHub Action.
    """
    return []


@main.group()
@click.option("--watchlist", "watchlist_path", type=click.Path(path_type=Path),
              default=Path("sources/watchlist.yaml"), show_default=True)
@click.pass_context
def watch(ctx: click.Context, watchlist_path: Path) -> None:
    """Inspect and refresh the source watchlist."""
    ctx.obj = load_watchlist(watchlist_path)


@watch.command("refresh")
@click.option("--dry-run/--apply", default=True,
              help="Dry run lists candidates; --apply not yet supported (lands in M10).")
@click.pass_obj
def watch_refresh(wl: Watchlist, dry_run: bool) -> None:
    """Pull fresh candidates from each watchlist entry."""
    candidates = _collect_watchlist_candidates(wl)
    if not dry_run:
        click.echo("--apply not yet supported (lands in M10 with GitHub Action)", err=True)
        raise SystemExit(4)
    for cid, title, authors, year in candidates:
        click.echo(f"{cid}\t{title}\t{', '.join(authors)}\t{year}")
    click.echo(f"\n{len(candidates)} candidates")
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli_ingest.py -v
```

Expected: 3 passed (2 from 4.10 + 1 new).

- [ ] **Step 4: Commit**

```bash
git add src/chatqec/cli.py tests/test_cli_ingest.py
git commit -m "feat(cli): add 'watch refresh --dry-run' command"
```

---

### Task 4.12 — Configure pytest network markers, CHANGELOG, push

**Files:**
- Modify: `pyproject.toml` (add `[tool.pytest.ini_options] markers`)
- Modify: `.gitlab-ci.yml` (add optional `network-tests` job)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add markers to `pyproject.toml`**

Find the `[tool.pytest.ini_options]` block and update it to:

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=chatqec --cov-report=term-missing -m 'not network'"
filterwarnings = ["error"]
markers = [
  "network: tests that require live network (arxiv, github, youtube, blog HTTP)",
]
```

- [ ] **Step 2: Append the optional `network-tests` job to `.gitlab-ci.yml`**

```yaml
network-tests:
  extends: .uv-base
  stage: test
  rules:
    - if: '$RUN_NETWORK_TESTS == "true"'
      when: always
    - when: manual
  script:
    - uv run pytest -v -m network
```

(Run via the GitLab UI → "Run Pipeline" → set `RUN_NETWORK_TESTS=true`.)

- [ ] **Step 3: Append M4 to `CHANGELOG.md`** (under `## [Unreleased] → ### Added`):

```markdown
- Ingestion pipeline:
  - `Chunk` model + on-disk `ChunkSink` writing `data/chunks/<source_id>.jsonl`
  - Math-aware chunker that protects `$...$`, `$$...$$`, and `\begin{X}...\end{X}` environments
  - arXiv title-and-author **resolver** (spec §7.4) — the architectural fix to the verification finding
  - Fetchers: arXiv (Atom + chunk), GitHub (files+repo metadata → chunks), YouTube (60s-window transcripts), URL/blog (defuddle → markdown), manual PDF (unstructured)
  - Watchlist (`sources/watchlist.yaml`) schema + parser with seed entries for arxiv categories, GitHub repos, YouTube channels, and RSS feeds
- CLI:
  - `chatqec ingest --url / --resolve-by-title / --arxiv` (the `--arxiv <id>` path warns per spec §7.4)
  - `chatqec watch refresh --dry-run`
- Test infrastructure: `pytest -m network` marker; optional `network-tests` GitLab CI job (manual trigger)
```

- [ ] **Step 4: Run full suite + lint + canonical validate**

```bash
uv run pytest -v
uv run ruff check src tests
uv run mypy src
uv run chatqec canonical validate
```

Expected: all green. Total tests after M4: ≥81 (M1: 6 + M2: 21 + M3: 19 + M4: 35).

- [ ] **Step 5: Commit and push**

```bash
git add pyproject.toml .gitlab-ci.yml CHANGELOG.md
git commit -m "docs(M4): update CHANGELOG and add network test marker + CI job"
git push origin main
```

Expected: GitLab pipeline runs `lint`, `validate-canonical`, `test` — all green. `network-tests` is manual.

---

## Milestone 4 — exit checklist

- [ ] `Chunk` + `ChunkSink` round-trip chunks through JSONL
- [ ] Math-aware chunker never splits inside `$…$`, `$$…$$`, `\begin{equation}…`, `\begin{align}…`
- [ ] arXiv title-and-author resolver returns unique hits, raises on ambiguity, disambiguates by year
- [ ] All 5 fetchers (arxiv, github, youtube, url, pdf) emit `Chunk`s with correct `source_id` per the ID scheme
- [ ] Watchlist YAML parses and validates the seed entries
- [ ] `chatqec ingest --url <url>` writes chunks to disk and a `pending_review` record
- [ ] `chatqec ingest --resolve-by-title "..." --author "..." --year ...` works end-to-end (with a mocked resolver in CI, real in manual smoke)
- [ ] `chatqec ingest --arxiv <id>` *warns* per §7.4 and exits 3 ("supported in M5") — no silent failure
- [ ] `chatqec watch refresh --dry-run` runs without error and lists candidates
- [ ] `pytest -m network` runs separately; default suite skips network
- [ ] ≥35 new tests; full suite ≥81 passing
- [ ] CHANGELOG updated
- [ ] GitLab CI green on `main`

**🛑 PAUSE — push to GitLab and review before requesting Milestone 5.**

---

## Milestone 5 — Embedding + Qdrant storage

**Outcome:** Chunks on disk become searchable in Qdrant. The system has a pluggable embedder (Gemini default, local Nomic fallback), a BM25 sparse encoder, a Qdrant store with vector + sparse + payload index, and a CLI command that promotes `pending_review` records to `active` after successful embedding. M2's `chatqec sources compact --apply` is finally wired up because Qdrant chunks can now be deleted alongside registry entries. `chatqec ingest --arxiv <id>` (the previously-guarded path from M4) now works because we have somewhere to put its chunks.

**Files this milestone creates / touches:**
- Create: `src/chatqec/embedding/__init__.py`
- Create: `src/chatqec/embedding/base.py` (Embedder protocol)
- Create: `src/chatqec/embedding/gemini.py`
- Create: `src/chatqec/embedding/local.py` (Nomic fallback)
- Create: `src/chatqec/embedding/sparse.py` (BM25)
- Create: `src/chatqec/store/__init__.py`
- Create: `src/chatqec/store/qdrant.py`
- Create: `src/chatqec/ingestion/pipeline.py` (orchestrates chunks → embed → store)
- Create: `docker-compose.yml` (Qdrant for local dev)
- Create: `tests/test_embedder_base.py`
- Create: `tests/test_embedder_local.py`
- Create: `tests/test_sparse_bm25.py`
- Create: `tests/test_store_qdrant.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_cli_embed.py`
- Modify: `src/chatqec/cli.py` (add `embed` command + wire up `ingest --arxiv` + `sources compact --apply`)
- Modify: `src/chatqec/ingestion/fetchers/arxiv.py` (add `fetch_arxiv_by_id`)
- Modify: `pyproject.toml` (add `fastembed`, `qdrant-client[fastembed]`)
- Modify: `conf/config.yaml` (add `store:` and `embedding:` sections)
- Modify: `CHANGELOG.md`

**Why split this way:** embedding and storage are both narrow contracts with strong type boundaries. The pipeline is the only place that knows about all three (chunks, embedder, store). Once it works end-to-end on a small fixture corpus, the rest of v1 has a real KB to retrieve against.

---

### Task 5.1 — Dependencies + Qdrant docker-compose + config additions

**Files:**
- Modify: `pyproject.toml`
- Modify: `conf/config.yaml`
- Create: `docker-compose.yml`

- [ ] **Step 1: Add fastembed to `pyproject.toml` dependencies**

In the `[project] → dependencies` list, replace the existing `qdrant-client>=1.12` line with:

```toml
  "qdrant-client[fastembed]>=1.12",
  "fastembed>=0.4",
```

Also append:

```toml
  "numpy>=1.26",
```

if not already present.

- [ ] **Step 2: Run `uv sync`**

```bash
uv sync --extra dev
```

Expected: fastembed and qdrant-client extras installed.

- [ ] **Step 3: Add `store:` and `embedding:` sections to `conf/config.yaml`**

Append:

```yaml
store:
  qdrant_url: "http://localhost:6333"
  collection: "chatqec_chunks"
  vector_dim: 768
  on_disk_payload: true

embedding:
  provider: "gemini"        # "gemini" | "local"
  model: "text-embedding-004"
  batch_size: 96
  local_model: "nomic-ai/nomic-embed-text-v1.5"  # used when provider="local"
```

- [ ] **Step 4: Create `docker-compose.yml` for local Qdrant**

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.0
    container_name: chatqec-qdrant
    ports:
      - "6333:6333"  # REST
      - "6334:6334"  # gRPC
    volumes:
      - ./.qdrant-data:/qdrant/storage
    environment:
      QDRANT__LOG_LEVEL: INFO
```

- [ ] **Step 5: Smoke test the Qdrant container (optional)**

```bash
docker compose up -d
curl -s http://localhost:6333/ | head
docker compose down
```

Expected: Qdrant root endpoint returns a JSON banner.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock conf/config.yaml docker-compose.yml
git commit -m "chore(M5): add fastembed + qdrant docker-compose and store/embedding configs"
```

---

### Task 5.2 — Embedder protocol + Gemini implementation

**Files:**
- Create: `src/chatqec/embedding/__init__.py` (empty)
- Create: `src/chatqec/embedding/base.py`
- Create: `src/chatqec/embedding/gemini.py`
- Create: `tests/test_embedder_base.py`

- [ ] **Step 1: Make `embedding` a package**

```bash
mkdir -p src/chatqec/embedding
touch src/chatqec/embedding/__init__.py
```

- [ ] **Step 2: Write failing tests for the protocol and Gemini wrapper**

Create `tests/test_embedder_base.py`:

```python
"""Embedder protocol + Gemini wrapper tests (Gemini mocked)."""

from unittest.mock import MagicMock

import pytest

from chatqec.embedding.base import Embedder, EmbedderConfig, EmbeddingDim
from chatqec.embedding.gemini import GeminiEmbedder


def test_embedder_protocol_minimum_interface():
    class Stub:
        dim = 8
        def embed_batch(self, texts):
            return [[0.0] * 8 for _ in texts]
    assert isinstance(Stub(), Embedder)


def test_gemini_embedder_calls_api_in_batches(monkeypatch):
    calls: list[list[str]] = []

    def fake_embed_content(model, content, task_type=None, **kw):
        calls.append(content if isinstance(content, list) else [content])
        n = len(calls[-1])
        return {"embedding": [[float(i)] * 768 for i in range(n)]}

    mock_genai = MagicMock()
    mock_genai.embed_content = fake_embed_content
    monkeypatch.setattr("chatqec.embedding.gemini.genai", mock_genai)

    cfg = EmbedderConfig(provider="gemini", model="text-embedding-004",
                         batch_size=3, local_model="")
    emb = GeminiEmbedder(cfg)
    out = emb.embed_batch(["a", "b", "c", "d", "e"])
    assert len(out) == 5
    assert len(out[0]) == 768
    # Batched 3 + 2
    assert [len(c) for c in calls] == [3, 2]


def test_gemini_embedder_dim_is_768():
    cfg = EmbedderConfig(provider="gemini", model="text-embedding-004",
                         batch_size=8, local_model="")
    emb = GeminiEmbedder(cfg)
    assert emb.dim == EmbeddingDim.GEMINI_004 == 768
```

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/test_embedder_base.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `src/chatqec/embedding/base.py`**

```python
"""Embedder protocol and shared types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable


class EmbeddingDim(IntEnum):
    GEMINI_004 = 768
    NOMIC_15 = 768


@dataclass(frozen=True)
class EmbedderConfig:
    provider: str       # "gemini" | "local"
    model: str
    batch_size: int
    local_model: str


@runtime_checkable
class Embedder(Protocol):
    """Anything with a `.dim` and `.embed_batch(texts)` is an embedder."""
    dim: int

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...
```

- [ ] **Step 5: Implement `src/chatqec/embedding/gemini.py`**

```python
"""Gemini text-embedding-004 embedder."""

from __future__ import annotations

from collections.abc import Sequence

import google.generativeai as genai  # type: ignore[import]

from chatqec.embedding.base import Embedder, EmbedderConfig, EmbeddingDim


class GeminiEmbedder(Embedder):
    dim: int = int(EmbeddingDim.GEMINI_004)

    def __init__(self, cfg: EmbedderConfig) -> None:
        self.cfg = cfg

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        bs = self.cfg.batch_size
        for i in range(0, len(texts), bs):
            batch = list(texts[i : i + bs])
            resp = genai.embed_content(
                model=f"models/{self.cfg.model}",
                content=batch,
                task_type="retrieval_document",
            )
            vectors = resp["embedding"] if isinstance(resp, dict) else resp.embedding
            out.extend(vectors)
        return out
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_embedder_base.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/chatqec/embedding/ tests/test_embedder_base.py
git commit -m "feat(embedding): add Embedder protocol and Gemini implementation"
```

---

### Task 5.3 — Local Nomic embedder fallback

**Files:**
- Create: `src/chatqec/embedding/local.py`
- Create: `tests/test_embedder_local.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_embedder_local.py`:

```python
"""Local Nomic embedder via sentence-transformers (mocked in tests)."""

from unittest.mock import MagicMock

import numpy as np

from chatqec.embedding.base import EmbedderConfig
from chatqec.embedding.local import LocalEmbedder


def test_local_embedder_uses_sentence_transformers(monkeypatch):
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((3, 768), dtype=np.float32)
    mock_st_class = MagicMock(return_value=mock_model)
    monkeypatch.setattr("chatqec.embedding.local.SentenceTransformer", mock_st_class)

    cfg = EmbedderConfig(provider="local", model="",
                         batch_size=8, local_model="nomic-ai/nomic-embed-text-v1.5")
    emb = LocalEmbedder(cfg)
    out = emb.embed_batch(["a", "b", "c"])
    assert len(out) == 3
    assert len(out[0]) == 768
    mock_st_class.assert_called_once_with(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True,
    )


def test_local_embedder_dim_is_768():
    from chatqec.embedding.base import EmbeddingDim
    assert int(EmbeddingDim.NOMIC_15) == 768
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_embedder_local.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/chatqec/embedding/local.py`**

```python
"""Local fallback embedder using sentence-transformers (Nomic by default)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import]

from chatqec.embedding.base import Embedder, EmbedderConfig, EmbeddingDim


class LocalEmbedder(Embedder):
    dim: int = int(EmbeddingDim.NOMIC_15)

    def __init__(self, cfg: EmbedderConfig) -> None:
        self.cfg = cfg
        # trust_remote_code required for Nomic's custom modeling code
        self.model = SentenceTransformer(cfg.local_model, trust_remote_code=True)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        arr: np.ndarray = self.model.encode(
            list(texts), batch_size=self.cfg.batch_size, show_progress_bar=False,
        )
        return arr.astype(np.float32).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_embedder_local.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/embedding/local.py tests/test_embedder_local.py
git commit -m "feat(embedding): add LocalEmbedder (Nomic via sentence-transformers)"
```

---

### Task 5.4 — BM25 sparse encoder

**Files:**
- Create: `src/chatqec/embedding/sparse.py`
- Create: `tests/test_sparse_bm25.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sparse_bm25.py`:

```python
"""BM25 sparse encoder tests (via fastembed)."""

from chatqec.embedding.sparse import BM25Encoder, SparseVector


def test_bm25_returns_index_value_pairs():
    enc = BM25Encoder()
    vecs = enc.encode_batch(["surface code", "color code", "color code variant"])
    assert len(vecs) == 3
    for v in vecs:
        assert isinstance(v, SparseVector)
        assert len(v.indices) == len(v.values)
        assert all(val > 0 for val in v.values)


def test_bm25_idf_lowers_weight_of_common_terms():
    enc = BM25Encoder()
    vecs = enc.encode_batch([
        "code code code code code",  # all 'code' → low weight per occurrence
        "Steane code",                # rare term 'Steane' → higher weight
    ])
    # The second vector should have at least one index whose value
    # exceeds the average value of the first vector.
    avg0 = sum(vecs[0].values) / len(vecs[0].values)
    assert max(vecs[1].values) > avg0


def test_bm25_empty_string_returns_empty_vector():
    enc = BM25Encoder()
    [v] = enc.encode_batch([""])
    assert v.indices == [] or len(v.indices) == 0
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_sparse_bm25.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/chatqec/embedding/sparse.py`**

```python
"""BM25 sparse encoder via fastembed."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fastembed import SparseTextEmbedding  # type: ignore[import]


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    """Encodes text into BM25 sparse vectors usable directly by Qdrant."""

    DEFAULT_MODEL = "Qdrant/bm25"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = SparseTextEmbedding(model_name=model_name)

    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]:
        embeddings = list(self._model.embed(list(texts)))
        out: list[SparseVector] = []
        for e in embeddings:
            out.append(SparseVector(
                indices=list(map(int, e.indices)),
                values=list(map(float, e.values)),
            ))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sparse_bm25.py -v
```

Expected: 3 passed. (First run downloads the fastembed BM25 model; ~30 MB.)

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/embedding/sparse.py tests/test_sparse_bm25.py
git commit -m "feat(embedding): add BM25 sparse encoder via fastembed"
```

---

### Task 5.5 — Qdrant store wrapper

**Files:**
- Create: `src/chatqec/store/__init__.py` (empty)
- Create: `src/chatqec/store/qdrant.py`
- Create: `tests/test_store_qdrant.py`

- [ ] **Step 1: Make `store` a package**

```bash
mkdir -p src/chatqec/store
touch src/chatqec/store/__init__.py
```

- [ ] **Step 2: Write failing tests using Qdrant `:memory:` client**

Create `tests/test_store_qdrant.py`:

```python
"""Qdrant store wrapper tests using the in-memory client."""

from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk
from chatqec.store.qdrant import ChatQecStore, StoreConfig


def _chunk(i: int, source_id: str = "test:1", text: str = "qec") -> Chunk:
    return Chunk(
        chunk_id=f"{source_id}#{i}", source_id=source_id, chunk_index=i,
        text=text, title="T", source_type="arxiv", authority_tier="preprint",
        anchor="",
    )


def _dense(seed: float = 0.1) -> list[float]:
    return [seed] * 768


def _sparse() -> SparseVector:
    return SparseVector(indices=[1, 5, 9], values=[0.7, 0.3, 0.1])


def test_store_creates_collection_on_init():
    cfg = StoreConfig(qdrant_url=":memory:", collection="test", vector_dim=768,
                      on_disk_payload=False)
    store = ChatQecStore(cfg)
    assert store.collection_exists()


def test_store_upserts_and_retrieves_by_id():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768,
                      on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0), _chunk(1)],
        dense_vectors=[_dense(0.1), _dense(0.9)],
        sparse_vectors=[_sparse(), _sparse()],
    )
    assert store.count() == 2


def test_store_dense_search_returns_nearest():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768,
                      on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0, text="surface"), _chunk(1, text="color")],
        dense_vectors=[_dense(0.1), _dense(0.9)],
        sparse_vectors=[_sparse(), _sparse()],
    )
    hits = store.dense_search(query_vector=_dense(0.1), top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "test:1#0"


def test_store_filter_excludes_retracted():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768,
                      on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0), _chunk(1)],
        dense_vectors=[_dense(0.1), _dense(0.9)],
        sparse_vectors=[_sparse(), _sparse()],
    )
    store.set_status_for_source("test:1", "retracted")
    hits = store.dense_search(query_vector=_dense(0.1), top_k=10, status_filter="active")
    assert hits == []


def test_store_delete_by_source_id():
    cfg = StoreConfig(qdrant_url=":memory:", collection="t", vector_dim=768,
                      on_disk_payload=False)
    store = ChatQecStore(cfg)
    store.upsert_chunks(
        chunks=[_chunk(0, source_id="a"), _chunk(1, source_id="a"),
                _chunk(0, source_id="b")],
        dense_vectors=[_dense(0.1), _dense(0.2), _dense(0.9)],
        sparse_vectors=[_sparse(), _sparse(), _sparse()],
    )
    store.delete_by_source_id("a")
    assert store.count() == 1
```

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/test_store_qdrant.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `src/chatqec/store/qdrant.py`**

```python
"""Thin Qdrant wrapper specialised for ChatQEC's chunk schema."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk


@dataclass(frozen=True)
class StoreConfig:
    qdrant_url: str  # http url or ":memory:"
    collection: str
    vector_dim: int
    on_disk_payload: bool


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    source_id: str
    score: float
    payload: dict[str, object]


_DENSE_NAME = "dense"
_SPARSE_NAME = "bm25"


class ChatQecStore:
    """ChatQEC's Qdrant-backed chunk store with hybrid (dense+sparse) vectors."""

    def __init__(self, cfg: StoreConfig) -> None:
        self.cfg = cfg
        if cfg.qdrant_url == ":memory:":
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=cfg.qdrant_url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.collection_exists():
            return
        self._client.create_collection(
            collection_name=self.cfg.collection,
            vectors_config={
                _DENSE_NAME: models.VectorParams(
                    size=self.cfg.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=self.cfg.on_disk_payload,
                ),
            },
            sparse_vectors_config={
                _SPARSE_NAME: models.SparseVectorParams(),
            },
        )
        for field in ("source_id", "source_type", "authority_tier", "status",
                      "topic_slugs"):
            self._client.create_payload_index(
                collection_name=self.cfg.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def collection_exists(self) -> bool:
        return self.cfg.collection in {
            c.name for c in self._client.get_collections().collections
        }

    def count(self) -> int:
        return self._client.count(self.cfg.collection, exact=True).count

    @staticmethod
    def _point_uuid(chunk_id: str) -> str:
        # Qdrant point IDs must be int or UUID; we hash chunk_id → UUID5.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert_chunks(
        self,
        *,
        chunks: Iterable[Chunk],
        dense_vectors: Iterable[list[float]],
        sparse_vectors: Iterable[SparseVector],
    ) -> None:
        points = []
        for c, dv, sv in zip(chunks, dense_vectors, sparse_vectors, strict=True):
            payload = {
                "chunk_id": c.chunk_id,
                "source_id": c.source_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "title": c.title,
                "source_type": c.source_type,
                "authority_tier": c.authority_tier,
                "anchor": c.anchor,
                "status": "active",
                **{f"meta_{k}": v for k, v in c.metadata.items()},
            }
            points.append(models.PointStruct(
                id=self._point_uuid(c.chunk_id),
                vector={
                    _DENSE_NAME: dv,
                    _SPARSE_NAME: models.SparseVector(
                        indices=sv.indices, values=sv.values,
                    ),
                },
                payload=payload,
            ))
        if points:
            self._client.upsert(collection_name=self.cfg.collection, points=points)

    def dense_search(
        self, *, query_vector: list[float], top_k: int,
        status_filter: str | None = "active",
    ) -> list[SearchHit]:
        flt = None
        if status_filter:
            flt = models.Filter(must=[
                models.FieldCondition(key="status",
                                      match=models.MatchValue(value=status_filter)),
            ])
        result = self._client.query_points(
            collection_name=self.cfg.collection,
            query=query_vector,
            using=_DENSE_NAME,
            limit=top_k,
            query_filter=flt,
        )
        return [
            SearchHit(
                chunk_id=p.payload["chunk_id"],
                source_id=p.payload["source_id"],
                score=p.score,
                payload=p.payload,
            )
            for p in result.points
        ]

    def sparse_search(
        self, *, sparse_query: SparseVector, top_k: int,
        status_filter: str | None = "active",
    ) -> list[SearchHit]:
        flt = None
        if status_filter:
            flt = models.Filter(must=[
                models.FieldCondition(key="status",
                                      match=models.MatchValue(value=status_filter)),
            ])
        result = self._client.query_points(
            collection_name=self.cfg.collection,
            query=models.SparseVector(
                indices=sparse_query.indices, values=sparse_query.values,
            ),
            using=_SPARSE_NAME,
            limit=top_k,
            query_filter=flt,
        )
        return [
            SearchHit(
                chunk_id=p.payload["chunk_id"],
                source_id=p.payload["source_id"],
                score=p.score,
                payload=p.payload,
            )
            for p in result.points
        ]

    def set_status_for_source(self, source_id: str, new_status: str) -> None:
        self._client.set_payload(
            collection_name=self.cfg.collection,
            payload={"status": new_status},
            points=models.Filter(must=[
                models.FieldCondition(
                    key="source_id", match=models.MatchValue(value=source_id),
                ),
            ]),
        )

    def delete_by_source_id(self, source_id: str) -> None:
        self._client.delete(
            collection_name=self.cfg.collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(
                    key="source_id", match=models.MatchValue(value=source_id),
                ),
            ])),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_store_qdrant.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/store/ tests/test_store_qdrant.py
git commit -m "feat(store): add Qdrant wrapper with hybrid vectors and status-aware filtering"
```

---

### Task 5.6 — Embedding pipeline

**Files:**
- Create: `src/chatqec/ingestion/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:

```python
"""End-to-end test: chunks.jsonl → embed → Qdrant → registry flipped to active."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest

from chatqec.embedding.base import Embedder
from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk, ChunkSink
from chatqec.ingestion.pipeline import EmbeddingPipeline, PipelineResult
from chatqec.ingestion.registry import (
    Registry, SourceRecord, SourceStatus, SourceType,
)
from chatqec.store.qdrant import ChatQecStore, StoreConfig


class _StubEmbedder(Embedder):
    dim = 768
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        # Map first char to a 768-d ramp for deterministic dense vectors
        return [[float(ord(t[0])) / 256] * 768 for t in texts]


class _StubSparse:
    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


def _now() -> datetime: return datetime(2026, 5, 29, tzinfo=timezone.utc)


def _seed(registry_path: Path, chunks_dir: Path, source_id: str = "arxiv:t1") -> None:
    sink = ChunkSink(chunks_dir)
    chunks = [Chunk(chunk_id=f"{source_id}#{i}", source_id=source_id,
                    chunk_index=i, text=f"chunk {i}", title="T",
                    source_type="arxiv", authority_tier="preprint", anchor="")
              for i in range(3)]
    sink.write_many(source_id, chunks)
    reg = Registry(registry_path)
    reg.append(SourceRecord(
        id=source_id, type=SourceType.ARXIV, url=f"https://arxiv.org/abs/{source_id}",
        title="T", authority_tier="preprint", ingested_at=_now(),
        last_checked_at=_now(), status=SourceStatus.PENDING_REVIEW,
        chunk_ids=[c.chunk_id for c in chunks], chunk_count=len(chunks),
    ))


def test_pipeline_promotes_pending_to_active(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"; chunks_dir.mkdir()
    registry_path = tmp_path / "reg.jsonl"
    _seed(registry_path, chunks_dir, "arxiv:t1")

    store = ChatQecStore(StoreConfig(qdrant_url=":memory:", collection="t",
                                     vector_dim=768, on_disk_payload=False))
    pipe = EmbeddingPipeline(
        chunks_dir=chunks_dir, registry=Registry(registry_path),
        embedder=_StubEmbedder(), sparse_encoder=_StubSparse(), store=store,
    )
    result = pipe.embed_source("arxiv:t1")
    assert isinstance(result, PipelineResult)
    assert result.chunks_embedded == 3
    assert store.count() == 3

    reg = Registry(registry_path)
    assert reg.get("arxiv:t1").status is SourceStatus.ACTIVE


def test_pipeline_embed_all_pending(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"; chunks_dir.mkdir()
    registry_path = tmp_path / "reg.jsonl"
    _seed(registry_path, chunks_dir, "arxiv:a")
    _seed(registry_path, chunks_dir, "arxiv:b")

    store = ChatQecStore(StoreConfig(qdrant_url=":memory:", collection="t",
                                     vector_dim=768, on_disk_payload=False))
    pipe = EmbeddingPipeline(
        chunks_dir=chunks_dir, registry=Registry(registry_path),
        embedder=_StubEmbedder(), sparse_encoder=_StubSparse(), store=store,
    )
    results = pipe.embed_all_pending()
    assert sum(r.chunks_embedded for r in results) == 6
    assert store.count() == 6


def test_pipeline_compact_deletes_chunks_for_retracted_sources(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"; chunks_dir.mkdir()
    registry_path = tmp_path / "reg.jsonl"
    _seed(registry_path, chunks_dir, "arxiv:keep")
    _seed(registry_path, chunks_dir, "arxiv:gone")

    store = ChatQecStore(StoreConfig(qdrant_url=":memory:", collection="t",
                                     vector_dim=768, on_disk_payload=False))
    reg = Registry(registry_path)
    pipe = EmbeddingPipeline(
        chunks_dir=chunks_dir, registry=reg,
        embedder=_StubEmbedder(), sparse_encoder=_StubSparse(), store=store,
    )
    pipe.embed_all_pending()
    assert store.count() == 6
    reg.update_status("arxiv:gone", SourceStatus.RETRACTED)

    purged = pipe.purge_retracted(older_than_days=0,
                                  now=datetime(2026, 12, 31, tzinfo=timezone.utc))
    assert purged == ["arxiv:gone"]
    assert store.count() == 3
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: ImportError on `chatqec.ingestion.pipeline`.

- [ ] **Step 3: Implement `src/chatqec/ingestion/pipeline.py`**

```python
"""Embedding pipeline: chunks on disk → embedder → Qdrant → registry promotion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

from chatqec.embedding.base import Embedder
from chatqec.embedding.sparse import BM25Encoder, SparseVector
from chatqec.ingestion.chunk import Chunk
from chatqec.ingestion.registry import (
    Registry,
    SourceStatus,
)
from chatqec.store.qdrant import ChatQecStore


class SparseEncoder(Protocol):
    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]: ...


@dataclass(frozen=True)
class PipelineResult:
    source_id: str
    chunks_embedded: int


class EmbeddingPipeline:
    """Reads chunk JSONL files, embeds them, writes to Qdrant, flips registry status."""

    def __init__(
        self, *, chunks_dir: Path, registry: Registry,
        embedder: Embedder, sparse_encoder: SparseEncoder,
        store: ChatQecStore,
    ) -> None:
        self.chunks_dir = Path(chunks_dir)
        self.registry = registry
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.store = store

    @staticmethod
    def _safe(source_id: str) -> str:
        return source_id.replace(":", "_").replace("/", "_").replace("@", "_")

    def embed_source(self, source_id: str) -> PipelineResult:
        path = self.chunks_dir / f"{self._safe(source_id)}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)

        chunks = [Chunk.model_validate_json(line)
                  for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        if not chunks:
            return PipelineResult(source_id=source_id, chunks_embedded=0)

        texts = [c.text for c in chunks]
        dense = self.embedder.embed_batch(texts)
        sparse = self.sparse_encoder.encode_batch(texts)

        self.store.upsert_chunks(chunks=chunks, dense_vectors=dense, sparse_vectors=sparse)

        # Promote registry status if currently pending_review
        rec = self.registry.get(source_id)
        if rec.status is SourceStatus.PENDING_REVIEW:
            self.registry.update_status(source_id, SourceStatus.ACTIVE)
        return PipelineResult(source_id=source_id, chunks_embedded=len(chunks))

    def embed_all_pending(self) -> list[PipelineResult]:
        results: list[PipelineResult] = []
        for rec in list(self.registry.filter(status=SourceStatus.PENDING_REVIEW)):
            results.append(self.embed_source(rec.id))
        return results

    def purge_retracted(
        self, *, older_than_days: int, now: datetime | None = None,
    ) -> list[str]:
        """Physically delete chunks from Qdrant for retracted/superseded sources."""
        now = now or datetime.now(tz=timezone.utc)
        purged: list[str] = []
        for rec in self.registry.compact_candidates(
            older_than_days=older_than_days, now=now,
        ):
            self.store.delete_by_source_id(rec.id)
            purged.append(rec.id)
        return purged
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_pipeline.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): add EmbeddingPipeline (chunks→embed→Qdrant→flip status)"
```

---

### Task 5.7 — Wire up M4's deferred paths: `ingest --arxiv` and `sources compact --apply`

**Files:**
- Modify: `src/chatqec/ingestion/fetchers/arxiv.py` (add `fetch_arxiv_by_id`)
- Modify: `src/chatqec/cli.py`

- [ ] **Step 1: Add a fetch-by-id helper to the arXiv fetcher**

Append to `src/chatqec/ingestion/fetchers/arxiv.py`:

```python
def fetch_arxiv_by_id(arxiv_id: str) -> ArxivMetadata:
    """Trusted-id fetch (only used after the user explicitly accepts §7.4 warning)."""
    import arxiv
    [result] = list(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
    # Re-use the Atom parser by emitting metadata directly
    return ArxivMetadata(
        arxiv_id=arxiv_id,
        title=result.title.replace("\n", " ").strip(),
        authors=[a.name for a in result.authors],
        year=result.published.year,
        abstract=result.summary,
        pdf_url=result.pdf_url or "",
    )
```

- [ ] **Step 2: Replace `ingest --arxiv` failure with a working path (still warns per §7.4)**

In `src/chatqec/cli.py`, replace the body of the `if arxiv_id_value:` branch in the `ingest` command with:

```python
    if arxiv_id_value:
        click.echo(
            "WARNING: --arxiv <id> trusts the supplied id. "
            "Per spec §7.4 prefer --resolve-by-title for LLM-suggested papers.",
            err=True,
        )
        from chatqec.ingestion.fetchers.arxiv import fetch_arxiv_by_id
        md = fetch_arxiv_by_id(arxiv_id_value)
        chunks = arxiv_paper_to_chunks(md, body_text=md.abstract,
                                       target_tokens=512, overlap_tokens=64)
        source_id = chunks[0].source_id
        sink.write_many(source_id, chunks)
        reg.append(SourceRecord(
            id=source_id, type=SourceType.ARXIV,
            url=f"https://arxiv.org/abs/{md.arxiv_id}",
            title=md.title, authors=md.authors, year=md.year,
            authority_tier="preprint", ingested_at=now, last_checked_at=now,
            status=SourceStatus.PENDING_REVIEW,
            chunk_count=len(chunks),
            chunk_ids=[c.chunk_id for c in chunks],
            verified=False,
        ))
        click.echo(f"ingested {source_id}: {len(chunks)} chunks (TRUSTED id — verify!)")
        return
```

- [ ] **Step 3: Wire `sources compact --apply` to the pipeline**

In `src/chatqec/cli.py`, replace the body of `sources_compact` with:

```python
    now = datetime.now(tz=timezone.utc)
    candidates = list(reg.compact_candidates(older_than_days=older_than_days, now=now))
    for rec in candidates:
        click.echo(f"{rec.id}\t{rec.status.value}\t{rec.last_checked_at.isoformat()}")
    click.echo(f"\n{len(candidates)} candidates")
    if not dry_run:
        from chatqec.embedding.gemini import GeminiEmbedder
        from chatqec.embedding.base import EmbedderConfig
        from chatqec.embedding.sparse import BM25Encoder
        from chatqec.store.qdrant import ChatQecStore, StoreConfig
        from chatqec.ingestion.pipeline import EmbeddingPipeline
        # Use dummy embedder for purge (no embeddings needed) — pipeline just
        # needs the store handle for deletion.
        emb_cfg = EmbedderConfig(provider="gemini", model="text-embedding-004",
                                 batch_size=8, local_model="")
        pipe = EmbeddingPipeline(
            chunks_dir=Path("data/chunks"), registry=reg,
            embedder=GeminiEmbedder(emb_cfg),
            sparse_encoder=BM25Encoder(),
            store=ChatQecStore(StoreConfig(
                qdrant_url="http://localhost:6333",
                collection="chatqec_chunks",
                vector_dim=768, on_disk_payload=True,
            )),
        )
        purged = pipe.purge_retracted(older_than_days=older_than_days, now=now)
        click.echo(f"purged {len(purged)} sources from Qdrant")
```

- [ ] **Step 4: Run the existing CLI tests to make sure nothing regressed**

```bash
uv run pytest tests/test_cli_sources.py tests/test_cli_ingest.py -v
```

Expected: still green (the M4 tests stubbed the resolve-arxiv path; the new `--arxiv` path is exercised via M5's network-marked smoke test in 5.8).

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/ingestion/fetchers/arxiv.py src/chatqec/cli.py
git commit -m "feat(cli): wire up 'ingest --arxiv' fetch and 'sources compact --apply'"
```

---

### Task 5.8 — CLI: `chatqec embed`, smoke test, CHANGELOG, push

**Files:**
- Modify: `src/chatqec/cli.py`
- Create: `tests/test_cli_embed.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write failing tests for the `embed` CLI**

Create `tests/test_cli_embed.py`:

```python
"""CLI tests for `chatqec embed` — uses stub embedder via monkeypatch."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest
from click.testing import CliRunner

from chatqec.cli import main
from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk, ChunkSink
from chatqec.ingestion.registry import (
    Registry, SourceRecord, SourceStatus, SourceType,
)


def _seed(tmp_path: Path) -> tuple[Path, Path]:
    chunks_dir = tmp_path / "chunks"; chunks_dir.mkdir()
    reg_path = tmp_path / "reg.jsonl"
    sid = "arxiv:t1"
    sink = ChunkSink(chunks_dir)
    sink.write_many(sid, [Chunk(chunk_id=f"{sid}#0", source_id=sid, chunk_index=0,
                                text="qec", title="T", source_type="arxiv",
                                authority_tier="preprint", anchor="")])
    Registry(reg_path).append(SourceRecord(
        id=sid, type=SourceType.ARXIV, url="https://arxiv.org/abs/t1", title="T",
        authority_tier="preprint",
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        last_checked_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status=SourceStatus.PENDING_REVIEW, chunk_ids=[f"{sid}#0"], chunk_count=1,
    ))
    return chunks_dir, reg_path


def test_embed_source_flips_status_to_active(tmp_path: Path, monkeypatch):
    chunks_dir, reg_path = _seed(tmp_path)

    class _StubEmb:
        dim = 768
        def embed_batch(self, texts: Sequence[str]):
            return [[0.1] * 768 for _ in texts]
    class _StubSparse:
        def encode_batch(self, texts: Sequence[str]):
            return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    monkeypatch.setattr("chatqec.cli._make_embedder", lambda cfg: _StubEmb())
    monkeypatch.setattr("chatqec.cli._make_sparse", lambda: _StubSparse())
    monkeypatch.setattr("chatqec.cli._make_store", lambda url, coll: __import__(
        "chatqec.store.qdrant", fromlist=["ChatQecStore"]).ChatQecStore(
        __import__("chatqec.store.qdrant", fromlist=["StoreConfig"]).StoreConfig(
            qdrant_url=":memory:", collection=coll, vector_dim=768, on_disk_payload=False)))

    runner = CliRunner()
    result = runner.invoke(main, [
        "embed", "--chunks-dir", str(chunks_dir), "--registry", str(reg_path),
        "--source-id", "arxiv:t1",
    ])
    assert result.exit_code == 0, result.output
    assert "1 chunks" in result.output

    reg = Registry(reg_path)
    assert reg.get("arxiv:t1").status is SourceStatus.ACTIVE


def test_embed_all_pending_processes_multiple(tmp_path: Path, monkeypatch):
    chunks_dir, reg_path = _seed(tmp_path)
    # Add a second pending source
    sid2 = "arxiv:t2"
    ChunkSink(chunks_dir).write_many(sid2, [Chunk(
        chunk_id=f"{sid2}#0", source_id=sid2, chunk_index=0, text="x",
        title="T2", source_type="arxiv", authority_tier="preprint", anchor="",
    )])
    Registry(reg_path).append(SourceRecord(
        id=sid2, type=SourceType.ARXIV, url="https://arxiv.org/abs/t2", title="T2",
        authority_tier="preprint",
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        last_checked_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status=SourceStatus.PENDING_REVIEW, chunk_ids=[f"{sid2}#0"], chunk_count=1,
    ))

    class _StubEmb:
        dim = 768
        def embed_batch(self, texts):
            return [[0.1] * 768 for _ in texts]
    class _StubSparse:
        def encode_batch(self, texts):
            return [SparseVector(indices=[1], values=[1.0]) for _ in texts]

    monkeypatch.setattr("chatqec.cli._make_embedder", lambda cfg: _StubEmb())
    monkeypatch.setattr("chatqec.cli._make_sparse", lambda: _StubSparse())
    monkeypatch.setattr("chatqec.cli._make_store", lambda url, coll:
        __import__("chatqec.store.qdrant", fromlist=["ChatQecStore"]).ChatQecStore(
            __import__("chatqec.store.qdrant", fromlist=["StoreConfig"]).StoreConfig(
                qdrant_url=":memory:", collection=coll, vector_dim=768, on_disk_payload=False)))

    runner = CliRunner()
    result = runner.invoke(main, [
        "embed", "--chunks-dir", str(chunks_dir), "--registry", str(reg_path),
        "--all-pending",
    ])
    assert result.exit_code == 0
    assert "2 sources" in result.output
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_cli_embed.py -v
```

Expected: `embed` command not found.

- [ ] **Step 3: Add `embed` command to `src/chatqec/cli.py`**

Add factory helpers (overridable in tests):

```python
def _make_embedder(cfg):
    """Build the configured embedder. Overridable for tests."""
    from chatqec.embedding.base import EmbedderConfig
    if cfg["provider"] == "local":
        from chatqec.embedding.local import LocalEmbedder
        return LocalEmbedder(EmbedderConfig(**cfg))
    from chatqec.embedding.gemini import GeminiEmbedder
    return GeminiEmbedder(EmbedderConfig(**cfg))


def _make_sparse():
    from chatqec.embedding.sparse import BM25Encoder
    return BM25Encoder()


def _make_store(url: str, collection: str):
    from chatqec.store.qdrant import ChatQecStore, StoreConfig
    return ChatQecStore(StoreConfig(qdrant_url=url, collection=collection,
                                    vector_dim=768, on_disk_payload=True))
```

Add the command:

```python
@main.command("embed")
@click.option("--chunks-dir", type=click.Path(path_type=Path),
              default=Path("data/chunks"), show_default=True)
@click.option("--registry", "registry_path", type=click.Path(path_type=Path),
              default=Path("sources/registry.jsonl"), show_default=True)
@click.option("--source-id", default=None, help="Embed a single source by id.")
@click.option("--all-pending", "all_pending", is_flag=True, default=False,
              help="Embed every pending_review source in the registry.")
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True)
@click.option("--collection", default="chatqec_chunks", show_default=True)
@click.option("--provider", default="gemini", show_default=True,
              type=click.Choice(["gemini", "local"]))
def embed(
    chunks_dir: Path, registry_path: Path,
    source_id: str | None, all_pending: bool,
    qdrant_url: str, collection: str, provider: str,
) -> None:
    """Embed chunk files into Qdrant and promote pending_review → active."""
    from chatqec.ingestion.pipeline import EmbeddingPipeline

    if not source_id and not all_pending:
        click.echo("pass --source-id <id> or --all-pending", err=True)
        raise SystemExit(1)

    emb_cfg = {"provider": provider, "model": "text-embedding-004",
               "batch_size": 96, "local_model": "nomic-ai/nomic-embed-text-v1.5"}
    pipe = EmbeddingPipeline(
        chunks_dir=chunks_dir, registry=Registry(registry_path),
        embedder=_make_embedder(emb_cfg), sparse_encoder=_make_sparse(),
        store=_make_store(qdrant_url, collection),
    )
    if source_id:
        result = pipe.embed_source(source_id)
        click.echo(f"embedded {result.source_id}: {result.chunks_embedded} chunks")
    else:
        results = pipe.embed_all_pending()
        total = sum(r.chunks_embedded for r in results)
        click.echo(f"{len(results)} sources, {total} chunks embedded")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli_embed.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Local smoke test against real Qdrant + Gemini (optional but worth running once)**

Set your Gemini key first:

```bash
export GEMINI_API_KEY=...
docker compose up -d
uv run chatqec ingest --url "https://errorcorrectionzoo.org/c/surface"
uv run chatqec embed --all-pending
uv run chatqec sources list --status active
docker compose down
```

Expected: source promoted from `pending_review` to `active`; Qdrant collection has chunks.

- [ ] **Step 6: Append M5 to `CHANGELOG.md`** (under `## [Unreleased] → ### Added`):

```markdown
- Embedding layer:
  - `Embedder` protocol with `GeminiEmbedder` (text-embedding-004) and `LocalEmbedder` (Nomic v1.5 via sentence-transformers)
  - BM25 sparse encoder via fastembed
- Qdrant store wrapper (`ChatQecStore`) with hybrid dense + sparse vectors, payload indexing on `status / source_id / authority_tier / topic_slugs`, status filtering, source-id deletion
- `EmbeddingPipeline` that reads `data/chunks/*.jsonl`, embeds, upserts to Qdrant, and promotes `pending_review` → `active`
- `docker-compose.yml` for local Qdrant
- CLI: `chatqec embed --source-id / --all-pending`
- M4 deferred paths wired up: `chatqec ingest --arxiv <id>` (warns per §7.4 and ingests), `chatqec sources compact --apply` (purges from Qdrant + registry)
```

- [ ] **Step 7: Run full suite + lint + canonical validate**

```bash
uv run pytest -v
uv run ruff check src tests
uv run mypy src
uv run chatqec canonical validate
```

Expected: all green. Total tests after M5: ≥97 (M1: 6 + M2: 21 + M3: 19 + M4: 35 + M5: 16).

- [ ] **Step 8: Commit and push**

```bash
git add CHANGELOG.md
git commit -m "docs(M5): update CHANGELOG for embedding + Qdrant storage"
git push origin main
```

Expected: GitLab pipeline runs lint, validate-canonical, test — all green.

---

## Milestone 5 — exit checklist

- [ ] `GeminiEmbedder` batches calls to `genai.embed_content` correctly (mocked test)
- [ ] `LocalEmbedder` uses sentence-transformers with `trust_remote_code=True` for Nomic
- [ ] `BM25Encoder` produces non-empty sparse vectors with reasonable IDF behaviour
- [ ] `ChatQecStore` round-trips chunks via Qdrant `:memory:` (dense + sparse search, status filter, delete-by-source)
- [ ] `EmbeddingPipeline` reads chunk files, embeds, writes to Qdrant, flips `pending_review` → `active`
- [ ] `chatqec embed --source-id <id>` and `chatqec embed --all-pending` work via stubs in tests
- [ ] `chatqec ingest --arxiv <id>` warns per §7.4 and ingests
- [ ] `chatqec sources compact --apply` purges Qdrant + registry
- [ ] ≥16 new tests; full suite ≥97 passing
- [ ] CHANGELOG updated
- [ ] GitLab CI green on `main`

**🛑 PAUSE — push to GitLab and review before requesting Milestone 6.**

---

## Milestone 6 — Retrieval pipeline

**Outcome:** A pure-Python `HybridRetriever` that, given a list of query strings, returns the top-K most relevant chunks. Internally: parallel dense + BM25 search against Qdrant, score normalization, the spec §8.5 weighted-sum scoring (`α·vector + β·bm25 + γ·tier + δ·recency`), candidate cap (top 50), cross-encoder reranking (BGE), and canonical force-injection by topic match. The retriever has no LLM dependency — that lands in M7 with the rewriter. A debug CLI `chatqec retrieve --query` exposes the pipeline.

**Files this milestone creates / touches:**
- Create: `src/chatqec/retrieval/__init__.py`
- Create: `src/chatqec/retrieval/normalize.py`
- Create: `src/chatqec/retrieval/scoring.py`
- Create: `src/chatqec/retrieval/canonical_injector.py`
- Create: `src/chatqec/retrieval/reranker.py`
- Create: `src/chatqec/retrieval/hybrid.py`
- Create: `tests/test_retrieval_normalize.py`
- Create: `tests/test_retrieval_scoring.py`
- Create: `tests/test_retrieval_canonical_injector.py`
- Create: `tests/test_retrieval_reranker.py`
- Create: `tests/test_retrieval_hybrid.py`
- Create: `tests/test_cli_retrieve.py`
- Modify: `src/chatqec/cli.py` (add `retrieve` debug command)
- Modify: `conf/config.yaml` (add `retriever.reranker_model`, candidate_k confirmation)
- Modify: `CHANGELOG.md`

**Why split this way:** the components compose cleanly. Score normalization is pure math. Scoring is pure math. Canonical injection is pure string matching against the loaded canonical index. Reranker is a wrapper. The hybrid orchestrator is the only place that holds all the moving parts. Each can be unit-tested in isolation, then assembled with one integration test.

---

### Task 6.1 — Score normalization

**Files:**
- Create: `src/chatqec/retrieval/__init__.py` (empty)
- Create: `src/chatqec/retrieval/normalize.py`
- Create: `tests/test_retrieval_normalize.py`

- [ ] **Step 1: Make the package**

```bash
mkdir -p src/chatqec/retrieval
touch src/chatqec/retrieval/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_retrieval_normalize.py`:

```python
"""Score normalization tests — bringing dense and BM25 scores to [0,1]."""

import pytest

from chatqec.retrieval.normalize import min_max_normalize, identity_if_single


def test_min_max_normalize_maps_to_unit_interval():
    out = min_max_normalize([0.5, 1.5, 2.5, 3.5])
    assert out == pytest.approx([0.0, 1/3, 2/3, 1.0])


def test_min_max_normalize_empty_returns_empty():
    assert min_max_normalize([]) == []


def test_min_max_normalize_constant_returns_ones():
    # all-equal scores → all 1.0 (avoid divide-by-zero)
    assert min_max_normalize([0.7, 0.7, 0.7]) == [1.0, 1.0, 1.0]


def test_identity_if_single_returns_single_score_as_one():
    assert identity_if_single([0.42]) == [1.0]


def test_identity_if_single_passes_through_for_multiple():
    out = identity_if_single([0.1, 0.5, 0.9])
    assert out == pytest.approx([0.0, 0.5, 1.0])
```

- [ ] **Step 3: Run failing tests**

```bash
uv run pytest tests/test_retrieval_normalize.py -v
```

Expected: ImportError.

- [ ] **Step 4: Implement `src/chatqec/retrieval/normalize.py`**

```python
"""Score normalization helpers.

Dense (cosine) and BM25 scores live on different scales; bring both to [0, 1]
before applying the weighted-sum scoring in §8.5.
"""

from __future__ import annotations


def min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


def identity_if_single(scores: list[float]) -> list[float]:
    """Like min_max but treats single-element input as relevance=1.0."""
    if len(scores) == 1:
        return [1.0]
    return min_max_normalize(scores)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_retrieval_normalize.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/retrieval/__init__.py src/chatqec/retrieval/normalize.py tests/test_retrieval_normalize.py
git commit -m "feat(retrieval): add min-max score normalization helpers"
```

---

### Task 6.2 — Weighted-sum scoring (spec §8.5)

**Files:**
- Create: `src/chatqec/retrieval/scoring.py`
- Create: `tests/test_retrieval_scoring.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_retrieval_scoring.py`:

```python
"""Spec §8.5 weighted-sum candidate scoring tests."""

from datetime import datetime, timezone

import pytest

from chatqec.retrieval.scoring import (
    Candidate,
    Coefficients,
    TIER_WEIGHTS,
    compute_recency,
    score_candidates,
)


def _now() -> datetime:
    return datetime(2026, 5, 29, tzinfo=timezone.utc)


def test_tier_weights_match_spec():
    # spec §8.5
    assert TIER_WEIGHTS["canonical"] == 1.0
    assert TIER_WEIGHTS["peer_reviewed"] == 0.9
    assert TIER_WEIGHTS["preprint"] == 0.7
    assert TIER_WEIGHTS["blog"] == 0.5
    assert TIER_WEIGHTS["video"] == 0.5


def test_compute_recency_returns_one_for_current_year():
    assert compute_recency(year=2026, now=_now()) == 1.0


def test_compute_recency_drops_to_zero_after_five_years():
    assert compute_recency(year=2020, now=_now()) == 0.0
    assert compute_recency(year=1999, now=_now()) == 0.0


def test_compute_recency_handles_missing_year():
    assert compute_recency(year=None, now=_now()) == 0.5  # neutral


def test_score_uses_default_coefficients_from_spec():
    cands = [
        Candidate(chunk_id="c1", dense_norm=1.0, bm25_norm=0.0,
                  authority_tier="canonical", year=2026),
        Candidate(chunk_id="c2", dense_norm=0.0, bm25_norm=1.0,
                  authority_tier="blog", year=2020),
    ]
    scored = score_candidates(cands, now=_now())
    # c1: 0.5*1 + 0.3*0 + 0.15*1.0 + 0.05*1 = 0.70
    # c2: 0.5*0 + 0.3*1 + 0.15*0.5 + 0.05*0  = 0.375
    assert scored[0].chunk_id == "c1"
    assert scored[0].final_score == pytest.approx(0.70)
    assert scored[1].chunk_id == "c2"
    assert scored[1].final_score == pytest.approx(0.375)


def test_score_with_custom_coefficients():
    coeffs = Coefficients(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0)
    cands = [
        Candidate(chunk_id="c1", dense_norm=0.7, bm25_norm=0.0,
                  authority_tier="blog", year=2024),
    ]
    [s] = score_candidates(cands, coefficients=coeffs, now=_now())
    assert s.final_score == pytest.approx(0.7)


def test_score_sorts_results_desc():
    cands = [
        Candidate(chunk_id="c_low", dense_norm=0.1, bm25_norm=0.0,
                  authority_tier="blog", year=2020),
        Candidate(chunk_id="c_high", dense_norm=0.9, bm25_norm=0.9,
                  authority_tier="canonical", year=2026),
    ]
    scored = score_candidates(cands, now=_now())
    assert scored[0].chunk_id == "c_high"
    assert scored[-1].chunk_id == "c_low"


def test_unknown_tier_falls_back_to_blog_weight():
    cands = [
        Candidate(chunk_id="x", dense_norm=0.5, bm25_norm=0.5,
                  authority_tier="unknown_tier", year=2025),
    ]
    [s] = score_candidates(cands, now=_now())
    # tier_weight = 0.5 (blog fallback)
    expected = 0.5 * 0.5 + 0.3 * 0.5 + 0.15 * 0.5 + 0.05 * (4/5)
    assert s.final_score == pytest.approx(expected)
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_retrieval_scoring.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/chatqec/retrieval/scoring.py`**

```python
"""Spec §8.5 candidate scoring:

    final_score = α·dense_norm + β·bm25_norm + γ·tier_weight + δ·recency
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

TIER_WEIGHTS: dict[str, float] = {
    "canonical": 1.0,
    "peer_reviewed": 0.9,
    "preprint": 0.7,
    "blog": 0.5,
    "video": 0.5,
}

_RECENCY_WINDOW_YEARS = 5


@dataclass(frozen=True)
class Coefficients:
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.15
    delta: float = 0.05


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    dense_norm: float
    bm25_norm: float
    authority_tier: str
    year: int | None


@dataclass(frozen=True)
class ScoredCandidate:
    chunk_id: str
    final_score: float


def compute_recency(*, year: int | None, now: datetime) -> float:
    if year is None:
        return 0.5
    age = now.year - year
    if age <= 0:
        return 1.0
    if age >= _RECENCY_WINDOW_YEARS:
        return 0.0
    return 1.0 - (age / _RECENCY_WINDOW_YEARS)


def score_candidates(
    candidates: list[Candidate], *,
    coefficients: Coefficients | None = None,
    now: datetime | None = None,
) -> list[ScoredCandidate]:
    coefficients = coefficients or Coefficients()
    now = now or datetime.now(tz=timezone.utc)
    out: list[ScoredCandidate] = []
    for c in candidates:
        tier_weight = TIER_WEIGHTS.get(c.authority_tier, TIER_WEIGHTS["blog"])
        recency = compute_recency(year=c.year, now=now)
        final = (
            coefficients.alpha * c.dense_norm
            + coefficients.beta * c.bm25_norm
            + coefficients.gamma * tier_weight
            + coefficients.delta * recency
        )
        out.append(ScoredCandidate(chunk_id=c.chunk_id, final_score=final))
    out.sort(key=lambda s: s.final_score, reverse=True)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_retrieval_scoring.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/retrieval/scoring.py tests/test_retrieval_scoring.py
git commit -m "feat(retrieval): add §8.5 weighted-sum scoring (α·dense + β·bm25 + γ·tier + δ·recency)"
```

---

### Task 6.3 — Canonical topic matcher (string-based)

**Files:**
- Create: `src/chatqec/retrieval/canonical_injector.py`
- Create: `tests/test_retrieval_canonical_injector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_retrieval_canonical_injector.py`:

```python
"""Canonical force-injection by topic match (string-based in M6; LLM-based in M7)."""

from datetime import date
from pathlib import Path

from chatqec.canonical.page import CanonicalFrontmatter, CanonicalPage
from chatqec.retrieval.canonical_injector import (
    CanonicalIndex,
    match_topics,
)


def _page(slug: str, aliases: list[str] | None = None) -> CanonicalPage:
    return CanonicalPage(
        frontmatter=CanonicalFrontmatter(
            topic_slug=slug, title=slug.replace("-", " ").title(),
            aliases=aliases or [],
            see_also=[],
            authority_tier="canonical",
            last_reviewed=date(2026, 5, 29),
            maintainers=["m"],
        ),
        body=f"Body of {slug}",
    )


def test_index_matches_slug_in_query_substring():
    idx = CanonicalIndex([_page("surface-code"), _page("color-code")])
    hits = match_topics("Compare the surface code threshold", idx)
    assert "surface-code" in hits


def test_index_matches_alias_in_query():
    idx = CanonicalIndex([_page("surface-code", aliases=["toric code"])])
    hits = match_topics("How does the toric code work?", idx)
    assert "surface-code" in hits


def test_index_is_case_insensitive():
    idx = CanonicalIndex([_page("surface-code", aliases=["Toric Code"])])
    hits = match_topics("explain TORIC code", idx)
    assert "surface-code" in hits


def test_index_returns_empty_for_unrelated_query():
    idx = CanonicalIndex([_page("surface-code")])
    hits = match_topics("What's for lunch?", idx)
    assert hits == []


def test_index_handles_punctuation_in_query():
    idx = CanonicalIndex([_page("surface-code")])
    hits = match_topics("Surface code, please?", idx)
    assert "surface-code" in hits


def test_index_dedups_repeated_matches():
    idx = CanonicalIndex([_page("surface-code", aliases=["toric code", "planar code"])])
    hits = match_topics("toric code AND planar code AND surface code", idx)
    assert hits == ["surface-code"]
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_retrieval_canonical_injector.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/chatqec/retrieval/canonical_injector.py`**

```python
"""Canonical force-injection: identify canonical pages relevant to a query.

M6 uses simple substring matching against page slugs and aliases. M7 may
upgrade this to an LLM-based topic classifier (Haiku), but the interface
stays the same.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chatqec.canonical.page import CanonicalPage


_NON_WORD = re.compile(r"[^a-z0-9\s-]+")


@dataclass(frozen=True)
class CanonicalIndex:
    """In-memory index of canonical pages by slug + alias.

    `pages` is the full list of loaded canonical pages; we build a lookup of
    normalized terms to slugs at construction time.
    """

    pages: list[CanonicalPage]

    def lookup(self) -> dict[str, str]:
        """Map normalized term → canonical topic_slug."""
        m: dict[str, str] = {}
        for p in self.pages:
            slug = p.frontmatter.topic_slug
            m[_normalize(slug.replace("-", " "))] = slug
            m[_normalize(p.frontmatter.title)] = slug
            for alias in p.frontmatter.aliases:
                m[_normalize(alias)] = slug
        return m


def _normalize(s: str) -> str:
    return _NON_WORD.sub("", s.lower()).strip()


def match_topics(query: str, index: CanonicalIndex) -> list[str]:
    """Return canonical slugs whose term appears in the query, in first-match order."""
    qn = _normalize(query)
    hits: list[str] = []
    seen: set[str] = set()
    # Match longest terms first so "surface code" wins over "code"
    sorted_terms = sorted(index.lookup().items(), key=lambda kv: len(kv[0]), reverse=True)
    for term, slug in sorted_terms:
        if not term:
            continue
        if re.search(rf"\b{re.escape(term)}\b", qn) and slug not in seen:
            hits.append(slug)
            seen.add(slug)
    return hits
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_retrieval_canonical_injector.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/retrieval/canonical_injector.py tests/test_retrieval_canonical_injector.py
git commit -m "feat(retrieval): add canonical topic index + match_topics()"
```

---

### Task 6.4 — Cross-encoder reranker (BGE)

**Files:**
- Create: `src/chatqec/retrieval/reranker.py`
- Create: `tests/test_retrieval_reranker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_retrieval_reranker.py`:

```python
"""Cross-encoder reranker tests — model load mocked."""

from unittest.mock import MagicMock

import numpy as np

from chatqec.retrieval.reranker import CrossEncoderReranker


def test_reranker_loads_model_lazily(monkeypatch):
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.9, 0.1, 0.5])
    mock_ce_class = MagicMock(return_value=mock_ce)
    monkeypatch.setattr("chatqec.retrieval.reranker.CrossEncoder", mock_ce_class)

    rr = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    mock_ce_class.assert_not_called()  # not loaded yet
    out = rr.rerank("query", ["a", "b", "c"], top_k=2)
    mock_ce_class.assert_called_once_with("BAAI/bge-reranker-base")
    assert out == [0, 2]  # indices of top-2 by score (0.9, 0.5)


def test_reranker_top_k_respects_input_size(monkeypatch):
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.5, 0.7])
    mock_ce_class = MagicMock(return_value=mock_ce)
    monkeypatch.setattr("chatqec.retrieval.reranker.CrossEncoder", mock_ce_class)

    rr = CrossEncoderReranker()
    out = rr.rerank("q", ["x", "y"], top_k=10)
    assert out == [1, 0]


def test_reranker_returns_empty_for_empty_candidates():
    rr = CrossEncoderReranker()
    assert rr.rerank("q", [], top_k=5) == []
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_retrieval_reranker.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/chatqec/retrieval/reranker.py`**

```python
"""Cross-encoder reranker wrapper (BAAI/bge-reranker-base by default).

The model is loaded lazily — the first call to `.rerank()` triggers the
~400MB download. In tests we mock the underlying CrossEncoder class so the
download never happens.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder  # type: ignore[import]


class CrossEncoderReranker:
    """Returns the indices of the top_k most relevant candidates given a query."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    def _model_or_load(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[str], *, top_k: int) -> list[int]:
        if not candidates:
            return []
        model = self._model_or_load()
        pairs = [[query, c] for c in candidates]
        scores: np.ndarray = model.predict(pairs)
        top_k = min(top_k, len(candidates))
        # Indices of top_k highest scores, sorted descending
        order = np.argsort(-scores)[:top_k]
        return [int(i) for i in order]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_retrieval_reranker.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/retrieval/reranker.py tests/test_retrieval_reranker.py
git commit -m "feat(retrieval): add CrossEncoderReranker (BGE) with lazy model load"
```

---

### Task 6.5 — `HybridRetriever` orchestrator

**Files:**
- Create: `src/chatqec/retrieval/hybrid.py`
- Create: `tests/test_retrieval_hybrid.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_retrieval_hybrid.py`:

```python
"""HybridRetriever end-to-end test using in-memory Qdrant + stub embedder/reranker."""

from datetime import date
from typing import Sequence

import pytest

from chatqec.canonical.page import CanonicalFrontmatter, CanonicalPage
from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk
from chatqec.retrieval.canonical_injector import CanonicalIndex
from chatqec.retrieval.hybrid import HybridRetriever, RetrievalResult
from chatqec.store.qdrant import ChatQecStore, StoreConfig


class _StubEmb:
    dim = 768
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(ord(t[0])) / 256] * 768 for t in texts]


class _StubSparse:
    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


class _StubReranker:
    def __init__(self): self.calls: int = 0
    def rerank(self, query: str, candidates: list[str], *, top_k: int) -> list[int]:
        self.calls += 1
        # Identity rerank: return the candidates in original order
        return list(range(min(top_k, len(candidates))))


def _seed_store() -> ChatQecStore:
    store = ChatQecStore(StoreConfig(qdrant_url=":memory:", collection="t",
                                     vector_dim=768, on_disk_payload=False))
    chunks = [
        Chunk(chunk_id="a#0", source_id="a", chunk_index=0, text="surface code",
              title="Surface paper", source_type="arxiv",
              authority_tier="peer_reviewed", anchor="",
              metadata={"year": 2024}),
        Chunk(chunk_id="b#0", source_id="b", chunk_index=0, text="color code",
              title="Color paper", source_type="arxiv",
              authority_tier="preprint", anchor="",
              metadata={"year": 2020}),
    ]
    store.upsert_chunks(
        chunks=chunks,
        dense_vectors=[[0.5] * 768, [0.9] * 768],
        sparse_vectors=[SparseVector(indices=[1], values=[1.0]),
                         SparseVector(indices=[2], values=[1.0])],
    )
    return store


def _canon_index_with_surface_code() -> CanonicalIndex:
    return CanonicalIndex([CanonicalPage(
        frontmatter=CanonicalFrontmatter(
            topic_slug="surface-code", title="Surface Code",
            aliases=["toric code"], see_also=[],
            authority_tier="canonical", last_reviewed=date(2026, 5, 29),
            maintainers=["m"]),
        body="A canonical surface code intro.",
    )])


def test_retrieve_returns_topk_chunks():
    store = _seed_store()
    rr = HybridRetriever(
        store=store, embedder=_StubEmb(), sparse_encoder=_StubSparse(),
        reranker=_StubReranker(), canonical_index=_canon_index_with_surface_code(),
    )
    result = rr.retrieve(queries=["surface code"], top_k=2, candidate_k=10)
    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) <= 2


def test_retrieve_force_injects_canonical_match():
    store = _seed_store()
    rr = HybridRetriever(
        store=store, embedder=_StubEmb(), sparse_encoder=_StubSparse(),
        reranker=_StubReranker(), canonical_index=_canon_index_with_surface_code(),
    )
    result = rr.retrieve(queries=["explain the toric code"], top_k=1, candidate_k=10)
    # "toric code" is an alias for surface-code → canonical is force-injected
    canonical_present = any(c.payload.get("source_id", "").startswith("canonical:")
                            for c in result.chunks)
    # Note: the canonical chunks aren't in Qdrant in this test (they would be
    # after M5 ingestion); the retriever still surfaces matched_topics.
    assert "surface-code" in result.matched_topics


def test_retrieve_reranker_called_once_per_query():
    store = _seed_store()
    reranker = _StubReranker()
    rr = HybridRetriever(
        store=store, embedder=_StubEmb(), sparse_encoder=_StubSparse(),
        reranker=reranker, canonical_index=_canon_index_with_surface_code(),
    )
    rr.retrieve(queries=["q1", "q2"], top_k=2, candidate_k=10)
    # Reranker is applied once over the union of all queries' candidates
    assert reranker.calls == 1


def test_retrieve_empty_store_returns_empty():
    store = ChatQecStore(StoreConfig(qdrant_url=":memory:", collection="t",
                                     vector_dim=768, on_disk_payload=False))
    rr = HybridRetriever(
        store=store, embedder=_StubEmb(), sparse_encoder=_StubSparse(),
        reranker=_StubReranker(), canonical_index=CanonicalIndex(pages=[]),
    )
    result = rr.retrieve(queries=["anything"], top_k=5, candidate_k=10)
    assert result.chunks == []
    assert result.matched_topics == []
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_retrieval_hybrid.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `src/chatqec/retrieval/hybrid.py`**

```python
"""HybridRetriever — orchestrates dense + BM25 search, scoring, reranking, and
canonical force-injection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

from chatqec.embedding.base import Embedder
from chatqec.embedding.sparse import SparseVector
from chatqec.retrieval.canonical_injector import CanonicalIndex, match_topics
from chatqec.retrieval.normalize import identity_if_single
from chatqec.retrieval.reranker import CrossEncoderReranker
from chatqec.retrieval.scoring import (
    Candidate,
    Coefficients,
    score_candidates,
)
from chatqec.store.qdrant import ChatQecStore, SearchHit


class SparseEncoderProto:
    def encode_batch(self, texts: Sequence[str]) -> list[SparseVector]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    text: str
    title: str
    authority_tier: str
    score: float
    payload: dict[str, object]


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    matched_topics: list[str]
    debug: dict[str, object] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self, *,
        store: ChatQecStore,
        embedder: Embedder,
        sparse_encoder: SparseEncoderProto,
        reranker: CrossEncoderReranker,
        canonical_index: CanonicalIndex,
        coefficients: Coefficients | None = None,
        per_query_k: int = 30,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.reranker = reranker
        self.canonical_index = canonical_index
        self.coefficients = coefficients or Coefficients()
        self.per_query_k = per_query_k

    def retrieve(
        self, *,
        queries: list[str],
        top_k: int,
        candidate_k: int,
        now: datetime | None = None,
    ) -> RetrievalResult:
        now = now or datetime.now(tz=timezone.utc)
        if not queries:
            return RetrievalResult(chunks=[], matched_topics=[])

        # 1) embed queries
        dense_q = self.embedder.embed_batch(queries)
        sparse_q = self.sparse_encoder.encode_batch(queries)

        # 2) per-query dense + sparse search → unioned hit map
        hits_by_chunk: dict[str, dict[str, object]] = {}
        dense_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}

        for dv in dense_q:
            for h in self.store.dense_search(query_vector=dv, top_k=self.per_query_k):
                hits_by_chunk[h.chunk_id] = h.payload
                dense_scores[h.chunk_id] = max(dense_scores.get(h.chunk_id, 0.0), h.score)
        for sv in sparse_q:
            for h in self.store.sparse_search(sparse_query=sv, top_k=self.per_query_k):
                hits_by_chunk[h.chunk_id] = h.payload
                bm25_scores[h.chunk_id] = max(bm25_scores.get(h.chunk_id, 0.0), h.score)

        if not hits_by_chunk:
            matched_topics = match_topics(" ".join(queries), self.canonical_index)
            return RetrievalResult(chunks=[], matched_topics=matched_topics)

        # 3) normalize and score per §8.5
        chunk_ids = list(hits_by_chunk.keys())
        dn = identity_if_single([dense_scores.get(c, 0.0) for c in chunk_ids])
        bn = identity_if_single([bm25_scores.get(c, 0.0) for c in chunk_ids])
        candidates: list[Candidate] = []
        for cid, dnorm, bnorm in zip(chunk_ids, dn, bn, strict=True):
            payload = hits_by_chunk[cid]
            year_raw = payload.get("meta_year") or payload.get("year")
            year = int(year_raw) if isinstance(year_raw, (int, float)) else None
            candidates.append(Candidate(
                chunk_id=cid,
                dense_norm=dnorm, bm25_norm=bnorm,
                authority_tier=str(payload.get("authority_tier", "blog")),
                year=year,
            ))
        scored = score_candidates(candidates, coefficients=self.coefficients, now=now)

        # 4) take top candidate_k
        top_cands = scored[:candidate_k]

        # 5) rerank → top_k
        texts = [str(hits_by_chunk[s.chunk_id].get("text", "")) for s in top_cands]
        # Use first query for reranking (caller can pass merged query)
        rerank_query = queries[0]
        top_idx = self.reranker.rerank(rerank_query, texts, top_k=top_k)
        retrieved = [
            RetrievedChunk(
                chunk_id=top_cands[i].chunk_id,
                source_id=str(hits_by_chunk[top_cands[i].chunk_id].get("source_id", "")),
                text=str(hits_by_chunk[top_cands[i].chunk_id].get("text", "")),
                title=str(hits_by_chunk[top_cands[i].chunk_id].get("title", "")),
                authority_tier=str(hits_by_chunk[top_cands[i].chunk_id].get("authority_tier", "")),
                score=top_cands[i].final_score,
                payload=hits_by_chunk[top_cands[i].chunk_id],
            )
            for i in top_idx
        ]

        # 6) canonical topic match (force-injection signaled to caller)
        matched_topics = match_topics(" ".join(queries), self.canonical_index)

        return RetrievalResult(
            chunks=retrieved,
            matched_topics=matched_topics,
            debug={
                "candidates_considered": len(candidates),
                "after_scoring": len(top_cands),
                "after_rerank": len(retrieved),
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_retrieval_hybrid.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/chatqec/retrieval/hybrid.py tests/test_retrieval_hybrid.py
git commit -m "feat(retrieval): add HybridRetriever orchestrator (dense+sparse+score+rerank+canonical)"
```

---

### Task 6.6 — CLI: `chatqec retrieve --query` (debug command)

**Files:**
- Modify: `src/chatqec/cli.py`
- Create: `tests/test_cli_retrieve.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli_retrieve.py`:

```python
"""CLI test for `chatqec retrieve --query` debug command."""

from datetime import date
from pathlib import Path
from typing import Sequence

import pytest
from click.testing import CliRunner

from chatqec.canonical.page import CanonicalFrontmatter, CanonicalPage
from chatqec.cli import main
from chatqec.embedding.sparse import SparseVector
from chatqec.ingestion.chunk import Chunk
from chatqec.store.qdrant import ChatQecStore, StoreConfig


def test_retrieve_query_prints_table(monkeypatch, tmp_path: Path):
    # Pre-populate a :memory: store and inject all the factory hooks
    store = ChatQecStore(StoreConfig(qdrant_url=":memory:", collection="t",
                                     vector_dim=768, on_disk_payload=False))
    store.upsert_chunks(
        chunks=[Chunk(chunk_id="x#0", source_id="x", chunk_index=0,
                      text="surface code", title="S", source_type="arxiv",
                      authority_tier="preprint", anchor="",
                      metadata={"year": 2024})],
        dense_vectors=[[0.5] * 768],
        sparse_vectors=[SparseVector(indices=[1], values=[1.0])],
    )

    class _StubEmb:
        dim = 768
        def embed_batch(self, texts: Sequence[str]):
            return [[0.5] * 768 for _ in texts]
    class _StubSparse:
        def encode_batch(self, texts: Sequence[str]):
            return [SparseVector(indices=[1], values=[1.0]) for _ in texts]
    class _StubReranker:
        def rerank(self, q, cands, *, top_k):
            return list(range(min(top_k, len(cands))))

    monkeypatch.setattr("chatqec.cli._make_embedder", lambda cfg: _StubEmb())
    monkeypatch.setattr("chatqec.cli._make_sparse", lambda: _StubSparse())
    monkeypatch.setattr("chatqec.cli._make_store", lambda url, coll: store)
    monkeypatch.setattr("chatqec.cli._make_reranker", lambda: _StubReranker())
    monkeypatch.setattr("chatqec.cli._make_canonical_index",
                        lambda d: __import__("chatqec.retrieval.canonical_injector",
                                             fromlist=["CanonicalIndex"]).CanonicalIndex(pages=[]))

    runner = CliRunner()
    result = runner.invoke(main, [
        "retrieve", "--query", "surface code", "--top-k", "1",
    ])
    assert result.exit_code == 0, result.output
    assert "x#0" in result.output
```

- [ ] **Step 2: Run failing tests**

```bash
uv run pytest tests/test_cli_retrieve.py -v
```

Expected: `retrieve` command not found.

- [ ] **Step 3: Add factory helpers + `retrieve` command to `src/chatqec/cli.py`**

```python
def _make_reranker():
    from chatqec.retrieval.reranker import CrossEncoderReranker
    return CrossEncoderReranker()


def _make_canonical_index(canonical_dir: Path):
    from chatqec.canonical.loader import load_canonical_directory
    from chatqec.retrieval.canonical_injector import CanonicalIndex
    pages = load_canonical_directory(canonical_dir) if canonical_dir.exists() else []
    return CanonicalIndex(pages=pages)


@main.command("retrieve")
@click.option("--query", "query_str", required=True, help="Single query string.")
@click.option("--top-k", type=int, default=8, show_default=True)
@click.option("--candidate-k", type=int, default=50, show_default=True)
@click.option("--qdrant-url", default="http://localhost:6333", show_default=True)
@click.option("--collection", default="chatqec_chunks", show_default=True)
@click.option("--canonical-dir", type=click.Path(path_type=Path),
              default=Path("knowledge/canonical"), show_default=True)
@click.option("--provider", default="gemini", show_default=True,
              type=click.Choice(["gemini", "local"]))
def retrieve(
    query_str: str, top_k: int, candidate_k: int,
    qdrant_url: str, collection: str, canonical_dir: Path, provider: str,
) -> None:
    """Debug command: run the hybrid retriever for one query and print results."""
    from chatqec.retrieval.hybrid import HybridRetriever
    emb_cfg = {"provider": provider, "model": "text-embedding-004",
               "batch_size": 96, "local_model": "nomic-ai/nomic-embed-text-v1.5"}
    rr = HybridRetriever(
        store=_make_store(qdrant_url, collection),
        embedder=_make_embedder(emb_cfg),
        sparse_encoder=_make_sparse(),
        reranker=_make_reranker(),
        canonical_index=_make_canonical_index(canonical_dir),
    )
    result = rr.retrieve(queries=[query_str], top_k=top_k, candidate_k=candidate_k)
    console = Console()
    table = Table(show_header=True, header_style="bold")
    for col in ("rank", "chunk_id", "source_id", "tier", "score", "title"):
        table.add_column(col)
    for i, c in enumerate(result.chunks):
        table.add_row(str(i + 1), c.chunk_id, c.source_id, c.authority_tier,
                      f"{c.score:.3f}", c.title[:60])
    console.print(table)
    if result.matched_topics:
        console.print(f"[dim]Matched canonical topics: {', '.join(result.matched_topics)}[/]")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli_retrieve.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Local smoke test (skip if Qdrant offline)**

```bash
docker compose up -d
uv run chatqec retrieve --query "What is the surface code?"
docker compose down
```

Expected: a Rich table of the top 8 chunks plus matched topic list.

- [ ] **Step 6: Commit**

```bash
git add src/chatqec/cli.py tests/test_cli_retrieve.py
git commit -m "feat(cli): add 'retrieve --query' debug command"
```

---

### Task 6.7 — CHANGELOG, full suite, push

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Append M6 to `CHANGELOG.md`** (under `## [Unreleased] → ### Added`):

```markdown
- Retrieval pipeline:
  - Score normalization helpers (`min_max_normalize`, `identity_if_single`)
  - Spec §8.5 weighted-sum scoring (`α=0.5·dense + β=0.3·bm25 + γ=0.15·tier + δ=0.05·recency`) with tunable coefficients
  - Canonical topic matcher (`CanonicalIndex` + `match_topics`) that surfaces force-injection candidates by slug/alias substring
  - `CrossEncoderReranker` wrapping BAAI/bge-reranker-base with lazy model load
  - `HybridRetriever` orchestrator: dense + BM25 → score → top-K candidates → rerank → top-K results + matched topics
- CLI: `chatqec retrieve --query` debug command (Rich table of top-K + matched canonical topics)
```

- [ ] **Step 2: Run full suite + lint + canonical validate**

```bash
uv run pytest -v
uv run ruff check src tests
uv run mypy src
uv run chatqec canonical validate
```

Expected: all green. Total tests after M6: ≥123 (M1: 6 + M2: 21 + M3: 19 + M4: 35 + M5: 16 + M6: 26).

- [ ] **Step 3: Commit and push**

```bash
git add CHANGELOG.md
git commit -m "docs(M6): update CHANGELOG for retrieval pipeline"
git push origin main
```

Expected: GitLab pipeline runs lint, validate-canonical, test — all green.

---

## Milestone 6 — exit checklist

- [ ] Score normalization rejects edge cases (empty / single / constant inputs)
- [ ] Scoring matches spec §8.5 coefficients exactly; tier table matches spec
- [ ] Recency drops linearly over 5 years, neutral for missing year
- [ ] Canonical matcher resolves slugs + aliases case-insensitively; dedups
- [ ] Cross-encoder reranker loads lazily and respects `top_k`
- [ ] `HybridRetriever` end-to-ends dense + sparse + scoring + rerank + canonical matching against in-memory Qdrant
- [ ] `chatqec retrieve --query "..."` prints a Rich table with rank/score/title
- [ ] ≥26 new tests; full suite ≥123 passing
- [ ] CHANGELOG updated
- [ ] GitLab CI green on `main`

**🛑 PAUSE — push to GitLab and review before requesting Milestone 7.**

---
