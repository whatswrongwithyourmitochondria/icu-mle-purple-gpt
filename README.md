# MLE-Bench Purple Agent (AgentBeats Research Track)

This repository contains a purple agent for the AgentBeats competition, focused on the MLE-bench Research Track: https://agentbeats.dev/agentbeater/mle-bench

MLE-bench evaluates how well AI agents perform real-world machine learning engineering by testing them on 75 Kaggle competitions spanning data preparation, model training, and experiment iteration. It measures end-to-end ML problem-solving against human leaderboard baselines, making it a strong benchmark for agents designed to operate like practical ML engineers.

This agent serves A2A requests, unpacks a competition bundle, runs a tree/panel search over candidate `solution.py` programs, and returns `submission.csv`.

Current reported result: this implementation achieved a top benchmark score of **0.83103** on **`spaceship-titanic`**.

## Scope and Positioning

- Target platform: AgentBeats purple-agent flow
- Task type: Kaggle-style ML engineering tasks from MLE-bench
- Interface: A2A HTTP server (`src/server.py`)
- Runtime model: panelized pass@K search + candidate review + final selection/blending

## What Is Novel in This Implementation

- Panelized pass@K seats: multiple independent search seats run in parallel with different seeds, temperatures, and dispositions, then merge globally.
- Tree search over code variants: each seat runs `draft -> improve/debug -> finalize` instead of one monolithic attempt.
- UCB (Upper Confidence Bound)-style branch selection: branch expansion uses exploration/exploitation tradeoff (inspired by MLEvolve, AIDE ML implementations, and Monte-Carlo graph/tree search ideas).
- Debug-first policy with bounded repair budget: broken branches are repaired early, with explicit caps.
- Runner-owned protocol split: holdout/CV protocol is prepared in the runner to reduce leakage-prone freedom in candidate code.
- Reviewer-gated reranking: top candidates are LLM-reviewed for suspicious/leaky patterns before final ranking.
- Fallback and anti-fake-success checks: detects trivial/submission-copy behavior and falls back safely when needed.
- Diversity + final blending: seat diversity is intentional, and top clean candidates are blended using holdout-weighted ensembling when possible.

### UCB Selection Details

In `src/mle_solver/tree/selector.py`, improve-step parent selection is branch-level UCB:

```text
UCB(branch i) = mean_reward_i + c * sqrt( ln(total_plays + 1) / (plays_i + 1) )
```

Where:

- `c = ucb_explore_c` from config (default `1.0`)
- `plays_i` = number of prior `improve` nodes in branch `i`
- `total_plays` = sum of `plays_i` over candidate branches
- `mean_reward_i` is min-max normalized branch score (not raw CV):

```text
score_i = best_valid_cv_in_branch_i             (or -cv for minimize metrics)
mean_reward_i = (score_i - min(score)) / (max(score) - min(score))
```

If all branches have the same score, denominator is set to `1.0`, so all `mean_reward_i = 0`.

Debug policy is applied before UCB: buggy nodes are repaired first (up to `max_debug_attempts_per_node`) unless the branch already has a valid candidate.

### Blending Coefficients

In `src/mle_solver/ensemble.py`, blend weights are derived from holdout scores, then normalized:

```text
baseline = min(valid_holdouts)   if maximize
baseline = max(valid_holdouts)   if minimize

raw_weight_i = max(holdout_i - baseline, 1e-6)         if maximize
raw_weight_i = max(baseline - holdout_i, 1e-6)         if minimize
raw_weight_i = 0.0                                     if holdout_i is missing

norm_weight_i = raw_weight_i / sum(raw_weight)
```

Edge behavior:

- If fewer than 2 valid holdout scores exist, all candidates get equal raw weight `1.0`.
- Candidates with missing files, unreadable CSVs, or column mismatches are skipped.
- At least 2 valid submissions are required to blend; otherwise the system falls back to the best single candidate.

Per-column blending:

- ID-like columns are copied from the reference file (must match across candidates).
- Binary columns use weighted vote with threshold `>= 0.5`.
- Numeric columns use weighted average.
- Non-numeric columns use weighted mode.

## Model, Provider, and Current Parameters

Current setup uses **OpenAI gpt-5.4** via the OpenAI API.
Draft and improve generation calls use `reasoning_effort: medium`.

```yaml
llm:
  model: gpt-5.4
  base_url: "https://api.openai.com/v1"
  api_key: ""
  temperature: 0.6
  max_tokens: 12000
  timeout: 600
  max_retries: 3

search:
  num_drafts: 2
  max_steps: 40
  max_parallel: 2
  pass_k: 3
  ucb_explore_c: 1.0
  max_debug_attempts_per_node: 2
  final_top_k: 3
  grace_seconds: 180
  holdout_fraction: 0.20
  n_folds: 3
  dispositions: []
  seat_temperatures: [0.5, 0.6, 0.9]
```

## Prerequisites

1. Clone this repository:

```bash
git clone --branch main https://github.com/whatswrongwithyourmitochondria/icu-mle-purple.git
cd icu-mle-purple
```

2. Clone MLE-bench as well (for local benchmark workflows and assets):

```bash
git clone https://github.com/openai/mle-bench.git
```

## Project Structure

```text
icu-mle-purple/
├─ .github/                      # CI/workflow configs
├─ src/                          # Runtime server + solver implementation
│  ├─ server.py                  # A2A HTTP server entrypoint
│  ├─ agent.py                   # Agent orchestration (task -> solver -> artifact)
│  ├─ executor.py                # Executes agent task flow
│  ├─ messenger.py               # Status/progress messaging helpers
│  └─ mle_solver/                # Core search-and-solve engine
│     ├─ config.py               # YAML/env config loading + validation
│     ├─ llm.py                  # LLM client wrapper
│     ├─ panel.py                # pass@K seat fan-out + merge
│     ├─ runner.py               # Top-level competition run pipeline
│     ├─ ensemble.py             # Final submission blending logic
│     ├─ agents/                 # LLM "roles" for code generation/evaluation
│     │  ├─ code_gen.py          # Draft/improve/debug code generation
│     │  ├─ parser.py            # Parse run outputs into structured scores
│     │  └─ reviewer.py          # Suspicion/leakage review before final ranking
│     ├─ exec/                   # Sandboxed execution utilities
│     │  ├─ interpreter.py       # Runs candidate solution.py
│     │  ├─ code_fix.py          # Quick auto-fixes for common code issues
│     │  └─ fake_success.py      # Detects trivial/fake successful submissions
│     ├─ prompts/                # Prompt templates for solver steps
│     │  ├─ system.py            # Shared system prompt
│     │  ├─ draft.py             # Draft prompt + dispositions
│     │  ├─ improve.py           # Improve prompt + hint policy
│     │  └─ debug.py             # Debug prompt templates
│     ├─ protocol/               # Runner-owned evaluation protocol
│     │  ├─ contract.py          # Infer task contract (metric/target/direction)
│     │  └─ splits.py            # Build dev/holdout split artifacts
│     └─ tree/                   # Search tree data structures + loop
│        ├─ node.py              # Candidate node representation
│        ├─ journal.py           # Run history/state tracking
│        ├─ selector.py          # UCB-based branch selection
│        ├─ ranking.py           # Final reranking/penalties
│        └─ loop.py              # draft -> improve/debug -> finalize loop
├─ tests/                        # Unit/integration tests
│  ├─ conftest.py                # Shared pytest fixtures/options
│  ├─ test_agent.py              # Agent/A2A surface tests
│  ├─ test_config.py             # Config parsing/validation tests
│  ├─ test_protocol.py           # Protocol/split logic tests
│  ├─ test_tree.py               # Tree loop/selection tests
│  ├─ test_parser.py             # Parser tests
│  ├─ test_prompts.py            # Prompt construction tests
│  ├─ test_reviewer.py           # Reviewer logic tests
│  └─ test_fake_success.py       # Fake-success detection tests
├─ test_assessment.py            # Local green-vs-purple assessment runner
├─ mle-solver.yaml               # Main solver config (model/search/exec)
├─ pyproject.toml                # Python project + dependencies
├─ Dockerfile                    # Container build/runtime definition
├─ amber-manifest.json5          # Agent manifest metadata
├─ uv.lock                       # Locked dependency graph
└─ README.md                     # Project documentation
```

## Running Locally

Use at least two terminals.

### Terminal 1: run first agent

```bash
cd icu-mle-purple
uv sync
uv run src/server.py --port 9009
```

### Terminal 2: run second agent (for local assessment scenarios)

```bash
cd icu-mle-purple
uv run src/server.py --port 9010
```

Agent card examples:

- `http://127.0.0.1:9009/.well-known/agent-card.json`
- `http://127.0.0.1:9010/.well-known/agent-card.json`

## Running with Docker

Use two terminals as well.

### Terminal 1: build and run container

```bash
cd icu-mle-purple
docker build -t my-agent .
docker run -p 9009:9009 my-agent
```

### Terminal 2: run tests against the containerized agent

```bash
cd icu-mle-purple
uv sync --extra test
uv run pytest --agent-url http://localhost:9009
```

## Testing

Run A2A conformance-style checks against a running agent.

### Terminal A: start agent (local or docker)

Local:

```bash
uv run src/server.py --port 9009
```

Docker:

```bash
docker run -p 9009:9009 my-agent
```

### Terminal B: run tests

```bash
cd icu-mle-purple
uv sync --extra test
uv run pytest --agent-url http://localhost:9009
```

Optional quick connectivity test:

```bash
uv run pytest tests/test_agent.py -q --agent-url http://localhost:9009
```

## Local Assessment Script

This repo includes `test_assessment.py` to run local green-vs-purple style assessment.

Example command (`spaceship-titanic`):

```bash
uv run test_assessment.py --green-port 9009 --purple-port 9010 --competition spaceship-titanic
```

## Environment

At minimum, set API credentials in `.env`:

- `OPENAI_API_KEY` (OpenAI provider)

The code also supports provider-based fallback env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) depending on configured `base_url`.

## References

- AgentBeats tutorial: https://docs.agentbeats.dev/tutorial/
- AgentBeats MLE-bench page: https://agentbeats.dev/agentbeater/mle-bench
- A2A protocol: https://a2a-protocol.org/latest/
- MLE-bench GitHub: https://github.com/openai/mle-bench
- MLE-bench paper (arXiv): https://arxiv.org/abs/2410.07095
- MLE-bench overview: https://openai.com/index/mle-bench/
- MLEvolve code: https://github.com/InternScience/MLEvolve
- AIDE ML code: https://github.com/WecoAI/aideml
- Monte-Carlo Graph Search talk: https://eleurent.github.io/monte-carlo-graph-search/paper/talk/talk.pdf
- Monte-Carlo Graph Search paper: https://proceedings.mlr.press/v129/leurent20a/leurent20a.pdf
- Monte-Carlo Graph Search supplementary: https://proceedings.mlr.press/v129/leurent20a/leurent20a-supp.pdf
