# Reproducing the ChatQEC evaluation

This page lists, in order, every command that produces a number, table, or quoted transcript in the
NAACL 2027 demo paper. Each run writes a JSON Lines file whose first line is a `_header` with its
provenance: chatQEC and tool-server commits (with a dirty flag), the benchmark pins and their SHA-256,
the gold file's SHA-256, the served model, the tool environment's package versions, Python, and GPUs.
Rows keep the verbatim answer and tool calls, so a run can be rescored without asking a model again.

Steps marked **TBD** are not final yet; do not quote numbers from them.

## 0. Environment

| Item | Value used |
|---|---|
| Hardware | one NVIDIA RTX 6000 Ada (48 GB) for the model, the reranker, and CUDA-Q; CPU for the other decoders |
| chatQEC | Python 3.11, `uv sync --extra dev` |
| Tool server | `~/chatqec-mcp-tools`, Python 3.13, installed as a uv tool (below) |
| vLLM | 0.32.0 in `~/venvs/llmserve` |
| Models | Muse-Glimmer-30B and gemma-4-31B-it, FP8, 32k context, served one at a time on GPU 0 |
| Qdrant | 1.12 (`docker compose up -d`) |

```bash
cd ~/chatQEC && uv sync --extra dev
docker compose up -d
uv tool install --force \
  "$HOME/chatqec-mcp-tools[lightstim,tsim,ibm,decoders,mle,cudaq,paulice,ft,deltakit,guppy,decoderbench,ftprimitive]"
```

The tool server's dependencies are not locked by `uv tool install`. Check that the installed versions
match the `tool_env` block of the run headers you want to reproduce (for example `qiskit-paulice` 0.3
breaks `ibm_add_pauli_checks`, which is why the tools repo pins it below 0.3).

## 1. Models (one GPU)

```bash
# Muse-Glimmer on GPU 0. 0.80 leaves ~11 GB for the reranker and CUDA-Q decoders.
GPU_MEM_UTIL=0.80 scripts/serve_local_models.sh muse
# Later, for Gemma 4: stop Muse first (kill the vLLM process AND its VLLM::EngineCore worker),
# then serve Gemma on the same GPU. Its 32k context needs 7.1 GiB of KV cache, hence 0.87.
GEMMA_GPU=0 GPU_MEM_UTIL=0.87 scripts/serve_local_models.sh gemma
```

`CHATQEC_LOCAL_MODEL=<served name>` (or `--model` on `eval bench run`) pins one model with no silent
fallback, so every row is attributable to the model in its header.

## 2. Benchmarks and gold

Pins are in `eval/benchmarks/pins.yaml`; `chatqec eval bench` refuses a checkout at another commit.

```bash
mkdir -p data/benchmarks && cd data/benchmarks
git clone https://github.com/satvikmaurya/decoder-bench.git decoder-bench
git -C decoder-bench checkout 9733a55a3a0e37c618294bd2ebbe0a8d4c334943
git clone https://github.com/ShuwenKan/FTPrimitiveBench.git ft-primitive-bench
git -C ft-primitive-bench checkout b75fbd1c514bc4fb7fcca75618a191e6b58fa7ed
cd ../..
```

- **decoder-bench** gold is its shipped `results/benchmark_results.csv`, copied verbatim to
  `eval/benchmarks/decoder_bench/benchmark_results.csv`, then audited (step 4).
- **FTPrimitiveBench** ships no results; gold is computed with the library itself:

```bash
cd ~/chatqec-mcp-tools && uv run --extra ftprimitive python \
    ~/chatQEC/scripts/benchmarks/ft_primitive_gold.py ~/chatQEC/eval/benchmarks/ft_primitive_bench/gold.jsonl
```

## 3. Benchmark runs

Tool calls are seeded; `CHATQEC_TOOL_CACHE_DIR` makes an identical seeded call return the stored
result, so every model is scored on the same tool output and slow decoders run once.

```bash
export CHATQEC_TOOL_CACHE_DIR=data/benchmarks/tool_cache CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
uv run chatqec eval bench run decoder-bench      --model muse-glimmer --out eval/results/decoder-bench_muse-glimmer_$TS.jsonl
uv run chatqec eval bench run ft-primitive-bench --model muse-glimmer --out eval/results/ft-primitive-bench_muse-glimmer_$TS.jsonl
# swap to Gemma (section 1), then the same two commands with --model gemma-4
```

Every finished item is appended and flushed at once, so a crash never loses finished items. To
continue every unfinished run with the settings recorded in its own header, detached from the
terminal:

```bash
scripts/resume_bench.sh --dry-run        # state of each run: OK / BUSY / WAIT (model not served) / DRIFT
scripts/resume_bench.sh                  # resume missing items
scripts/resume_bench.sh --retry-errors   # also re-ask errored items (rows kept in <run>.errors.jsonl)
```

It refuses to resume when answer-path code or the tool environment changed since the run started
(`--allow-drift` overrides, and the drift is written to `logs/bench_resume_notes.txt`).

## 4. decoder-bench gold audit

For every pair whose shipped gold lies outside chatQEC's 95% interval, rebuild the circuit with
decoder-bench's own code (circuit SHA-256 checked against the tool's) and decode it with its own
decoder class at 5x the item's shots and a different seed (4242), in decoder-bench's native env:

```bash
uv venv --python 3.11 data/benchmarks/decoder-bench-native
uv pip install --python data/benchmarks/decoder-bench-native \
    -r eval/benchmarks/decoder_bench/native-requirements.txt
data/benchmarks/decoder-bench-native/bin/python scripts/benchmarks/decoder_bench_gold_audit.py \
    --decoder-bench data/benchmarks/decoder-bench \
    --out eval/benchmarks/decoder_bench/gold_audit.jsonl \
    eval/results/decoder-bench_*.jsonl
```

**TBD:** how audited gold replaces shipped gold in scoring (then `chatqec eval bench rescore`).

## 5. ChatQEC versus the bare model

```bash
uv run python scripts/benchmarks/build_compare_set.py .   # writes eval/compare/prompts.jsonl
uv run chatqec eval bench run compare --model muse-glimmer --system chatqec --judge-model gemma-4 --out ...
uv run chatqec eval bench run compare --model muse-glimmer --system bare    --judge-model gemma-4 --out ...
```

The judge (definition and abstain items only) is always the other local model, so no model grades
itself. **TBD:** rebuild the set with audited decoder-bench gold first; final Gemini and Claude bare
runs happen once, on the tagged commit.

## 6. Retrieval QA set

**TBD:** `chatqec eval run --cases eval/qa_set.jsonl --judge-provider local --judge-model gemma-4 ...`,
three runs, plus ablations (no canonical injection, no reranker, dense only).

## 7. Reports

```bash
uv run chatqec eval bench report eval/results/decoder-bench_*.jsonl
```

## 8. Chat UI acceptance

Drives the real Streamlit Chat page with Playwright for the prompts in
`eval/ui_acceptance_prompts.json` (5 tool prompts and 3 no-tool controls):

```bash
uv run --with playwright python scripts/ui_acceptance.py --port 8501 --label muse \
    --executable-path ~/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome
```

(with an app started as `CHATQEC_PROVIDER=local CHATQEC_LOCAL_MODEL=muse-glimmer
CHATQEC_MCP_SERVER=chatqec-mcp-tools uv run streamlit run chatqec_app/streamlit_app.py`).
`scripts/run_ui_acceptance.sh` runs both models side by side and needs two GPUs.

## Known caveats

- Even at temperature 0, vLLM batching can change outputs; model behaviour (for example whether a
  tool is called) varies between runs.
- `ibm_add_pauli_checks` is stochastic unless a seed is passed.
- decoder-bench does not pin relay-bp or mwpf; the native env pins the versions the audit used.
