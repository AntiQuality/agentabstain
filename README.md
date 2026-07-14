<div align="center">
  <img src="assets/logo.png" width="130" alt="AgentAbstain logo">

  <h1>AgentAbstain: Do LLM Agents Know When Not to Act?</h1>

  <p>
    <a href="https://agentabstain.github.io"><img src="https://img.shields.io/badge/Website-agentabstain.github.io-4a5fc1" alt="Website"></a>
    <a href="https://huggingface.co/datasets/antiquality/agentabstain"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d" alt="Dataset"></a>
    <a href="https://arxiv.org/abs/2607.10059"><img src="https://img.shields.io/badge/arXiv-2607.10059-b31b1b" alt="arXiv"></a>
  </p>

  <p><i>The first systematic evaluation framework for agentic abstention:<br>the calibrated ability of tool-using LLM agents to recognize when <b>not</b> to act.</i></p>
</div>

---

## Overview

Most agent benchmarks reward getting the task done. AgentAbstain tests the opposite skill: knowing when to stop. When a request is vague, self-contradictory, or impossible with the tools on hand, an agent that plows ahead can do real and irreversible damage.

The framework has two components:

- **AgentAbstain**, the benchmark: **263 paired tasks** in **42 executable MCP sandbox environments**, built on an agent-native taxonomy of **8 abstention scenarios** spanning pre-execution and runtime triggers. Every should-act task ships with a should-abstain variant that differs by a single controlled perturbation, so no always-act or always-refuse policy can exceed 50% paired accuracy.
- **AbstainGen**, the pipeline: synthesizes the environments and task pairs end-to-end, validated by deterministic DAG replay and cross-family LLM critics. Three independent annotators rate 94 to 98% of sampled tasks as well-designed.

Evaluation crosses a deterministic **commit check** on the tool-call trace with an **LLM judge** on the terminal response, which isolates failure modes that neither signal catches alone, such as *post-hoc abstention*: the agent executes the irreversible action first and claims restraint afterwards.

<div align="center">
  <img src="assets/figure1.png" width="620" alt="One should-abstain task, four qualitatively different agent behaviors">
  <p><i>One should-abstain task, four qualitatively different behaviors. Only Successful Abstention, where the agent holds the critical action and surfaces the conflict, counts as correct.</i></p>
</div>

## Leaderboard

Across 17 frontier LLMs in 4 agent harnesses, the best agent reaches only **59.5% paired accuracy**, and abstention capability is largely independent of task-solving capability.

<div align="center">
  <img src="assets/ranking_bar.png" width="820" alt="Paired accuracy across 17 frontier LLMs">
</div>

| # | Model | Harness | Act | Abstain | Paired | CAR |
|--:|:------|:--------|----:|--------:|-------:|----:|
| 1 | Gemini 3.1 Pro | Google ADK | 90.5 | 65.4 | **59.5** | 65.7 |
| 2 | Claude Opus 4.7 | Claude SDK | 76.5 | **79.0** | 59.4 | **77.6** |
| 3 | Claude Sonnet 4.6 | Claude SDK | 83.1 | 66.4 | 53.4 | 65.4 |
| 4 | GPT-5.5 | OpenAI SDK | 87.4 | 61.1 | 52.5 | 59.8 |
| 5 | Claude Haiku 4.5 | Claude SDK | 80.5 | 65.6 | 49.7 | 61.8 |
| 6 | GPT-5 | OpenAI SDK | 74.6 | 69.8 | 49.6 | 66.5 |
| 7 | GPT-5.4 | OpenAI SDK | 76.9 | 67.8 | 48.7 | 64.0 |
| 8 | GLM-5 | OpenClaw | 82.5 | 61.8 | 47.8 | 59.1 |
| 9 | GPT-OSS 120B | OpenClaw | 78.3 | 59.5 | 46.2 | 58.2 |
| 10 | GPT-5.2 | OpenAI SDK | 74.7 | 63.1 | 42.9 | 59.2 |
| 11 | MiniMax M2.5 | OpenClaw | 83.8 | 50.1 | 41.9 | 49.6 |
| 12 | DeepSeek V3.2 | OpenClaw | 82.4 | 52.1 | 41.4 | 50.2 |
| 13 | GPT-5.1 | OpenAI SDK | 75.0 | 60.7 | 40.6 | 53.2 |
| 14 | Gemini 3 Flash | Google ADK | **91.7** | 43.6 | 39.7 | 43.4 |
| 15 | DeepSeek V4 Pro | OpenClaw | 87.0 | 42.8 | 36.9 | 42.3 |
| 16 | Kimi K2.5 | OpenClaw | 63.9 | 52.0 | 33.4 | 53.2 |
| 17 | GPT-4o | OpenAI SDK | 82.1 | 44.2 | 33.0 | 40.9 |

**Paired** (primary): the share of pairs where the model gets both the should-act and should-abstain variants right. **Act** and **Abstain**: per-side pass rates. **CAR** (Conditioned Abstention Rate): abstain accuracy restricted to pairs whose act side the model already solved, isolating restraint from raw capability. All numbers are macro-averaged over the 8 scenarios.

## Repository Layout

```
src/                          inference runtime
├── runtime/                  agent harness integrations (Claude SDK, OpenAI SDK, Google ADK, OpenClaw)
├── configs/                  17 model configs + task sets (tasks.yaml is the full benchmark)
├── scripts/                  run.sh, run_inference.py, resume_session.py
└── types/                    shared task and rollout types

eval/                         evaluation harness
├── evaluators/               commit check (deterministic) + LLM response judge
├── configs/default.yaml      judge configuration
├── runner.py                 per-model evaluation entry point
├── statistics/               analysis and figure scripts behind every figure in the paper
└── scripts/eval.sh           batch evaluation across models
```

## Getting Started

Python 3.10+ is expected, plus the SDKs of the harnesses you want to run (OpenAI Agents SDK, Anthropic Claude SDK, Google ADK). Provider credentials are read from the environment.

**Run inference.** Rollouts are written to `results/{provider}/{model}/`:

```bash
# all models of one provider family
bash src/scripts/run.sh claudesdk        # openaisdk | googleadk | claudesdk | all

# or one model against one task set
python -m src.scripts.run_inference \
    --runtime-config src/configs/claudesdk_claude-opus-4-7.yaml \
    --task-config    src/configs/tasks.yaml \
    --workers 4
```

**Run evaluation.** The commit check and the LLM judge score saved rollouts:

```bash
python -m eval.runner --provider claudesdk --model claude-opus-4-7

# or batch across models
bash eval/scripts/eval.sh
```

**Regenerate figures.** Every figure in the paper is produced by a script under `eval/statistics/` (for example `figure_ranking_bar.py`, `figure_category_difficulty.py`); each reads the evaluation outputs from your own runs.

## Dataset

The 263 task pairs and the 42 sandbox environments are hosted on [Hugging Face](https://huggingface.co/datasets/antiquality/agentabstain); task sets under `src/configs/` reference dataset task IDs. The AbstainGen generation pipeline is fully documented in the paper and intentionally not open-sourced, as a public generator would let benchmark-matched training data be synthesized at scale. Fresh evaluation rounds can be generated privately on demand, which keeps the benchmark resistant to training-data contamination.

## License

Code is released under the [MIT License](LICENSE). The [dataset](https://huggingface.co/datasets/antiquality/agentabstain) is released under CC BY 4.0.

## Citation

```bibtex
@misc{liu2026agentabstain,
  title  = {AgentAbstain: Do LLM Agents Know When Not to Act?},
  author = {Liu, Xun and Zhang, Yi Evie and Kasprova, Vira and Rabbani, Parisa and Zahraei, Pardis Sadat and Zhang, Tianyu and Ebrahimpour-Boroojeny, Ali and Chandrasekaran, Varun},
  year   = {2026},
  eprint = {2607.10059},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```