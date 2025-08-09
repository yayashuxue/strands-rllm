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

8. **Run browser evaluation.** The repository includes a working BrowserComp-style evaluation:

   ```bash
   # Run demo tasks (3 simple web browsing questions)
   python eval/run_browsercomp.py --data data/demo_tasks.jsonl --limit 3 --max_steps 5
   
   # Download and run full BrowserComp dataset (requires Kaggle API)
   python download_browsecomp.py
   python eval/run_browsercomp.py --data data/browsecomp_official.jsonl --limit 100 --max_steps 3
   ```

   The agent uses the **LocalChromiumBrowser** tool to actually browse websites and extract information. Set `OPENAI_API_KEY` environment variable for OpenAI model access.

9. **Generate custom test data.** Create realistic browsing tasks for testing:

   ```bash
   python prepare_real_browsecomp_data.py --output data/custom_tasks.jsonl --limit 10
   ```

10. **Benchmark other tasks.** The `eval/` directory contains placeholders for running benchmarks such as **GAIA** and **XBench**. When ready, load the benchmark tasks, feed them to rLLM's `AgentExecutionEngine`, and compute the evaluation metrics for your agent.

## Browser Agent Evaluation

The repository includes a fully working browser-enabled agent that can:
- Navigate to websites using LocalChromiumBrowser
- Extract information from web pages
- Answer questions based on real-time web content

### Quick Demo
```bash
# Activate environment
conda activate rllm

# Run 3 demo tasks (iPhone price, Tokyo population, movie release date)
python eval/run_browsercomp.py --data data/demo_tasks.jsonl --limit 3 --max_steps 5
```

### Configuration
- **Model**: Uses OpenAI GPT-4o-mini (set `OPENAI_API_KEY` environment variable)
- **Browser**: LocalChromiumBrowser (headless Chromium via Playwright)
- **Evaluation**: Normalized exact match scoring with URL canonicalization

### Data Files
- `data/demo_tasks.jsonl` - 3 sample browsing tasks for quick testing
- `data/browsecomp_official.jsonl` - Full BrowserComp dataset (download via script)
- Generated via `prepare_real_browsecomp_data.py` and `download_browsecomp.py`

## Notes

- This skeleton deliberately avoids modifying the Strands SDK itself. It shows how to **compose** a new agent by importing the SDK and adding your own memory implementation.
- rLLM is a framework‑agnostic RL engine. It can run arbitrary Python workflows as long as you provide a reward function and dataset. See the Agentica documentation for more details on how to define workflows and rewards.
- The workflow uses `gpt-4o-mini` as the default model. Once the macOS compatibility PR is merged, you can use any compatible OpenAI model.
- Browser evaluation requires installing `bedrock-agentcore` dependency for LocalChromiumBrowser tool.

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
