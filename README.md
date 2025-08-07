# Strands + rLLM Integration Skeleton

This repository provides a starting point for integrating the AWS **Strands** agent framework with the **rLLM** reinforcement learning framework. The goal of this project is to help you build and evaluate a _memory‑enabled language agent_ that can be trained and benchmarked using rLLM, starting with the baseline **mem0** memory and later swapping in a **MEM1**-style memory implementation.

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
│   ├── workflow.py            # Example evaluation loop using AgentExecutionEngine
│   ├── strands_agent_wrapper.py  # Wrapper to integrate Strands agent with rLLM
│   └── strands_env.py         # Environment for agent interaction
├── local_tokenizer/           # Local tokenizer files for rLLM
│   ├── chat_template.jinja    # Chat template for the tokenizer
│   └── merges.txt             # Tokenizer merges file
├── download_tokenizer.py      # Script to download and setup the local tokenizer
├── eval/                      # Placeholders for benchmark evaluation scripts
│   ├── __init__.py
│   ├── run_browsercomp.py
│   ├── run_gaia.py
│   └── run_xbench.py
└── requirements.txt           # Suggested Python dependencies
```

## How to use this skeleton

### **Option A: Using Git Submodules (Recommended)**

1. **Clone with submodules:**

   ```bash
   git clone --recursive https://github.com/yayashuxue/strands-rllm.git
   cd strands-rllm
   ```

2. **Setup the local tokenizer:**

   ```bash
   python download_tokenizer.py
   ```

3. **Create and activate a conda environment:**

   ```bash
   conda create -n rllm python=3.11
   conda activate rllm
   ```

4. **Install dependencies:**

   ```bash
   # Install Strands SDK and other dependencies
   pip install -r requirements.txt

   # Install rLLM from submodule
   pip install -e ./verl
   pip install -e ./rllm
   ```

### **Option B: Manual Installation**

1. **Clone without submodules:**

   ```bash
   git clone https://github.com/yayashuxue/strands-rllm.git
   cd strands-rllm
   ```

2. **Setup the local tokenizer:**

   ```bash
   python download_tokenizer.py
   ```

3. **Create and activate a conda environment:**

   ```bash
   conda create -n rllm python=3.11
   conda activate rllm
   ```

4. **Install dependencies:**

   ```bash
   # Install Strands SDK and other dependencies
   pip install -r requirements.txt

   # Install rLLM manually
   git clone https://github.com/rllm-org/rllm.git
   cd rllm
   pip install -e ./verl
   pip install -e .
   cd ..
   ```

**Note:** Python 3.11 is required for torch 2.7+ compatibility on macOS. GPU features are automatically excluded on macOS for compatibility.

5. **Install the main package:**

   ```bash
   pip install -e .
   ```

6. **Run the baseline agent:**

   ```bash
   python rllm_workflow/workflow.py
   ```

   This script instantiates the agent and runs a small set of dummy tasks to demonstrate how to connect the agent to rLLM's `AgentExecutionEngine`. Replace the dummy tasks with real benchmark tasks.

7. **Implement MEM1 memory.** If you wish to replace the baseline memory with a MEM1-style compact state, use the stub in `strands_agent/memory/mem1_stub.py` as a starting point. Implement the memory consolidation logic in the `recall` and `update` methods and modify `strands_agent/agent.py` to instantiate `Mem1Memory` instead of `Mem0Memory`.

8. **Benchmark the agent.** The `eval/` directory contains placeholders for running benchmarks such as **BrowserComp**, **GAIA**, and **XBench**. When ready, load the benchmark tasks, feed them to rLLM's `AgentExecutionEngine`, and compute the evaluation metrics for your agent.

## Notes

- This skeleton deliberately avoids modifying the Strands SDK itself. It shows how to **compose** a new agent by importing the SDK and adding your own memory implementation.
- rLLM is a framework‑agnostic RL engine. It can run arbitrary Python workflows as long as you provide a reward function and dataset. See the Agentica documentation for more details on how to define workflows and rewards.
- The workflow uses `gpt-4o-mini` as the default model. Once the macOS compatibility PR is merged, you can use any compatible OpenAI model.
- Benchmarks mentioned above (BrowserComp, GAIA, XBench) require downloading their datasets or using evaluation APIs. Refer to the corresponding papers and repositories for instructions.

## Managing rLLM Submodule

If you chose Option A (submodules), here are common operations:

### **Check Status**

```bash
git submodule status
```

### **Update rLLM**

```bash
cd rllm
git checkout main
git pull origin main
cd ..
git add rllm
git commit -m "Update rLLM submodule"
```

### **Switch to Specific Version**

```bash
cd rllm
git checkout <commit-hash>
cd ..
git add rllm
git commit -m "Update rLLM submodule to specific version"
```

**Note:** Currently pointing to a fork with macOS compatibility patches. Once the PR is merged, you can update to the official version.
