# Strands + rLLM Integration Skeleton

This repository provides a starting point for integrating the AWS **Strands** agent framework with the **rLLM** reinforcement learning framework.  The goal of this project is to help you build and evaluate a *memory‑enabled language agent* that can be trained and benchmarked using rLLM, starting with the baseline **mem0** memory and later swapping in a **MEM1**-style memory implementation.

## Repository layout

```
strands-rllm/
├── strands_agent/             # Code for the Strands-based agent
│   ├── __init__.py
│   ├── agent.py               # Builds a Strands agent using mem0 memory (baseline)
│   └── memory/
│       ├── __init__.py
│       ├── mem0_adapter.py    # Optional helper to wrap the default mem0 memory
│       └── mem1_stub.py       # Stub for integrating a MEM1-like memory
├── rllm_workflow/             # Scripts for running the agent via rLLM workflows
│   ├── __init__.py
│   └── workflow.py            # Example evaluation loop using AgentWorkflowEngine
├── eval/                      # Placeholders for benchmark evaluation scripts
│   ├── __init__.py
│   ├── run_browsercomp.py
│   ├── run_gaia.py
│   └── run_xbench.py
└── requirements.txt           # Suggested Python dependencies
```

## How to use this skeleton

1. **Install dependencies.**  This project depends on the `strands` and `rllm` packages, neither of which is currently on PyPI.  Clone and install them from their respective repositories before proceeding:

   ```bash
   # Clone this repository (the skeleton itself).
   git clone https://github.com/yayashuxue/strands-rllm.git

   # Install dependencies.  You must have the Strands SDK and rLLM installed.
   # See https://github.com/aws/strands and https://github.com/agentica-project/rllm for installation instructions.
   pip install -r requirements.txt
   ```

2. **Run the baseline agent.**  The `strands_agent/agent.py` module defines a function `build_agent()` that constructs a Strands agent using the default `mem0` memory tool.  You can run the agent through an rLLM evaluation loop using the `rllm_workflow/workflow.py` script:

   ```bash
   python rllm_workflow/workflow.py
   ```

   This script instantiates the agent and runs a small set of dummy tasks to demonstrate how to connect the agent to rLLM's `AgentWorkflowEngine`.  Replace the dummy tasks with real benchmark tasks.

3. **Implement MEM1 memory.**  If you wish to replace the baseline memory with a MEM1-style compact state, use the stub in `strands_agent/memory/mem1_stub.py` as a starting point.  Implement the memory consolidation logic in the `recall` and `update` methods and modify `strands_agent/agent.py` to instantiate `Mem1Memory` instead of `Mem0Memory`.

4. **Benchmark the agent.**  The `eval/` directory contains placeholders for running benchmarks such as **BrowserComp**, **GAIA**, and **XBench**.  When ready, load the benchmark tasks, feed them to rLLM's `AgentWorkflowEngine`, and compute the evaluation metrics for your agent.

## Notes

* This skeleton deliberately avoids modifying the Strands SDK itself.  It shows how to **compose** a new agent by importing the SDK and adding your own memory implementation.
* rLLM is a framework‑agnostic RL engine.  It can run arbitrary Python workflows as long as you provide a reward function and dataset.  See the Agentica documentation for more details on how to define workflows and rewards.
* Benchmarks mentioned above (BrowserComp, GAIA, XBench) require downloading their datasets or using evaluation APIs.  Refer to the corresponding papers and repositories for instructions.
