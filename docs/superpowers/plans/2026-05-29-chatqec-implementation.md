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
