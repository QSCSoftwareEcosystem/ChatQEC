# Integrate the vendor QEC tools into chatQEC

Date: 2026-10-26. Status: draft, awaiting approval.

## Intent

chatQEC users should be able to ask in chat for the QEC tools the vendors ship,
and get real results with citations. Two flows must work end to end:

1. Decoder comparison: "decode this circuit with several decoders and compare them."
2. IBM Pauli checks: "add spacetime Pauli checks to my circuit and show the overhead."

`notebooks/getting_started.ipynb` is the user-facing record. Every tool added
gets a section there, in the same change that adds the tool, so a new user can
read the notebook and see what chatQEC can do.

## Where the work lives

Tools are MCP tools in the sibling repo `~/chatqec-mcp-tools` (registered in
`src/chatqec_mcp_tools/server.py`, one `ToolSpec` each, heavy deps behind
optional extras). chatQEC itself changes only for prompts
(`src/chatqec/conversation/prompts.py`), tool-output rendering
(`chatqec_app/components/tool_output.py`), RAG sources, the notebook, and tests.
This split already exists and is kept.

## Scope, in build order

1. `decoder_compare`: add MWPF, MLE and table decoders (bloqade-decoders), and
   CUDA-Q sliding window and AI pre-decoder. Extends the existing tool, no new name.
2. `ibm_add_pauli_checks`: Qiskit Paulice, spacetime Pauli checks and postselected detection.
3. `ibm_ft_compile`: Qiskit 2.5 FT transpile (Clifford+T, PBC pass manager, gridsynth).
4. `deltakit_memory`: Deltakit 0.8 code generation, noise, local and cloud decoders, leakage-aware LCD.
5. `guppy_run`: Guppy + Selene (+ Guppy FT) simulation.
6. BB5, barbell and tile code parameters added to `code_params`, plus RAG sources for them.

Each item is its own commit (tool, tests, prompt text, notebook section together).

## Goals

- Each tool is callable from chat, returns structured output the UI renders, and
  degrades cleanly when its extra is missing (per-tool `ok=False` with an install hint,
  the pattern `decoder_compare` already uses).
- Each tool is verified through the real chat UI before it is called done, because
  an upstream gate (e.g. the off-topic classifier) can block a new capability that
  unit tests never see.
- Notebook sections run end to end and are committed executed.

## Non-goals

- No new hosted services. Tools run as local subprocesses like the existing ones.
- No GPU requirement. CUDA-Q items must report unavailable without a GPU.
- No reimplementation of vendor algorithms; chatQEC wraps the vendor SDKs.
- Real IBM or Quantinuum hardware submission for the new tools is out of scope here.

## Anti-goals

- A tool that exists in the registry but is refused by the router or classifier in chat.
- Heavy vendor imports that slow server start or crash it when an extra is absent.
- Silent fallbacks that return plausible but fake numbers.

## Constraints

- Testing uses local models only: Muse-Glimmer-30B, Gemma 4 fallback. No Claude or Gemini tokens.
- Work on `main`, commit after each finished task, no Co-Authored-By trailer lines
  from Claude, Conventional Commits.
- Python 3.11, `uv`. Dry-run resolution on 2026-10-26 succeeded for qiskit-paulice,
  deltakit, guppylang, selene-sim, mwpf, cudaq-qec-cu13, bloqade-decoders.
- The repo has uncommitted eval work (generate, judge, seeds). It must not be mixed
  into these commits.
- Intern-repo mistakes found along the way are reported, not silently fixed.

## Decided by the user

- Both flows (decoder comparison, Pauli checks) are required.
- The build order above.
- Notebook is populated incrementally as features land.

- Deltakit: local decoders only, nothing that costs money. Cloud decoders report "not configured".
- BB5, barbell, tile parameters: cite from vendor papers (search done 2026-10-26, list below).
  Values are copied from the papers, never computed or recalled.

## Candidate sources for item 6 (found by web search, IDs not yet checked against the papers)

- BB5: arXiv:2503.22071, "Quantum error correction for long chains of trapped ions" (Ye, Delfosse).
  Hardware demo: arXiv:2606.06455, "Breakeven demonstration of quantum low-density parity-check codes".
  IonQ authorship of either is not confirmed.
- Tile: arXiv:2504.09171, "Tile Codes: High-Efficiency Quantum Codes on a Lattice with Boundary" (IQM authors).
  Directional tile codes: arXiv:2606.11484 (per Error Correction Zoo).
- Barbell (IQM, announced June 2026): only press coverage found, no arXiv ID yet. Needs one more
  search before any numbers are used. Press figures are marketing claims, not citable values.

## Answered by the project (from the builder check)

- Two repos: tools live in `~/chatqec-mcp-tools` (branch `main`), chatQEC pins it by
  `uv tool install --force ~/chatqec-mcp-tools` (README). Each tool gets paired commits,
  one per repo, reinstall before testing in chat.
- Environment: dry-run on Python 3.11 of `ibm`, `decoders`, `tsim` extras plus qiskit-paulice,
  deltakit, guppylang, selene-sim resolves in one env with no conflicts. No per-tool venvs.
- GPU: this machine has 2x RTX 6000 Ada, so CUDA-Q items run for real here. On other
  machines they report unavailable (probe by import and device check).
- Flow 1 is already partly met: `decoder_compare` has pymatching, tesseract, bposd, bplsd,
  belief_find, cudaq_relay_bp, cudaq_bposd. Item 1 widens it, it is not a prerequisite.

## My calls (please check)

- Circuit input: Stim text or OpenQASM; convert to Qiskit for Paulice and IBM tools.
- "Overhead" for Pauli checks: added qubits, gate count, depth, and postselection discard rate.
- Acceptance: one fixed chat prompt per flow, run against Muse-Glimmer then Gemma 4.
  A local model not calling the tool is treated as a prompt or router bug to fix, not accepted.
- Uncommitted eval work: stage only the files each commit touches, no stash.
- Notebook sections for a tool that can be unavailable show the real output on this machine.

## Left to the builder (my calls, to be checked)

- Input and output schemas for each new tool, following existing `ToolSpec` style.
- Which extra each tool lives behind (new extras `paulice`, `deltakit`, `guppy`, or reuse `ibm`).
- Whether sliding window and the AI pre-decoder are new `decoder_compare` entries or options.
- Which papers and docs go into RAG for item 6.
