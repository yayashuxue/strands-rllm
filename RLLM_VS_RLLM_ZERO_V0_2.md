## 概览 Overview

- 本文客观对比本仓库的 `rllm` 与子模块 `_external/rllm-zero`（v0.2 分支）的架构与功能差异，覆盖组件、引擎、集成、训练与示例等方面。
- This document provides an objective comparison between local `rllm` and the submodule `_external/rllm-zero` (branch v0.2), covering components, engines, integrations, training, and examples.

## 目录结构与新增模块 Directory Structure and New Modules

- 共同基础 Common base:

  - `rllm/agents`, `rllm/environments`, `rllm/engine`, `rllm/trainer`, `rllm/parser`, `rllm/tools`（命名或细节可能存在差异）

- rllm-zero v0.2 的显著新增 Notable additions in rllm-zero v0.2:
  - `rllm/workflows/`：工作流抽象与终止处理；便于高层编排（multi-agent、评测、奖励）
  - `rllm/engine/agent_workflow_engine.py`：`AgentWorkflowEngine`，运行工作流并行、重试、Episode 输出与 veRL 转换
  - `rllm/integrations/strands.py`：`RLLMModel` + `StrandsAgent` 适配层，复用 `RolloutEngine` 并输出 rLLM 轨迹
  - `rllm/api/api_server.py`：示例 API server，内置默认模型配置与单/多轮工作流引擎
  - `rllm/db/episode_store.py`（若存在）：Episode 存储与检索（用于持久化评测结果）
  - `examples/strands/`：Strands 工作流示例（研究、多步、数学评测）

## 引擎对比 Engines

| 项目 Project   | 核心引擎 Core Engines                                         | 说明 Notes                                                                                                                                                      |
| -------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| rllm           | `AgentExecutionEngine`（含 `AsyncAgentExecutionEngine` 别名） | 面向标准 `BaseAgent`/`BaseEnv` 的并行异步采样器；管理 agent-env 生命周期、模型调用、环境步进、超时/截断、奖励聚合；支持 Text/Token/Step/Conversation 多返回模式 |
| rllm-zero v0.2 | `AgentExecutionEngine` + `AgentWorkflowEngine`                | 新增 `AgentWorkflowEngine`，面向 `Workflow` 抽象进行高层编排，内置重试与 Episode→veRL 转换，便于训练闭环与多 Agent Workflow                                     |

要点 Highlights:

- ExecutionEngine 更贴近标准 RL 采样（Agent/Env 形态）。
- WorkflowEngine 更利于工作流（Workflow 形态）评测/编排与训练数据转换。

## RolloutEngine 角色 Role of RolloutEngine

- 统一推理后端：封装 tokenizer、chat parser、推理端点（OpenAI/veRL），提供 `get_model_response` 与相关接口。
- 在 rllm-zero v0.2：
  - 工作流路径（Workflow/AgentWorkflowEngine）直接依赖 `RolloutEngine`。
  - Strands 适配通过 `integrations/strands.RLLMModel` 复用 `RolloutEngine` 完成推理与消息转换。
- 在 rllm：
  - 由 `AgentExecutionEngine` 直接经 `AsyncOpenAI`（openai 模式）或经 veRL Router（verl 模式）使用；同样依赖 Chat 模板解析。

## Workflow 与 Episode/Trajectory Workflow and Episode/Trajectory

| 维度 Dimension               | rllm                                            | rllm-zero v0.2                                                                                                             |
| ---------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 抽象 Abstraction             | Agent/Env 导向                                  | Workflow 导向（单/多 Agent）                                                                                               |
| 输出 Output                  | `Trajectory` 为主（或 Token/Step/Conversation） | `Episode`（可含多条命名 `Trajectory`）                                                                                     |
| 存储 Storage                 | 通常由上层自定义                                | 可选 `EpisodeStore`，集中存储与索引                                                                                        |
| 训练转换 Training conversion | 由 Trainer/上层处理                             | `execute_tasks_verl` + `_transform_results_for_verl` 直接生成 veRL `DataProto`（含 stepwise advantage、compact filtering） |

## Strands 集成 Strands Integration

| 维度 Dimension       | rllm                  | rllm-zero v0.2                                                                                  |
| -------------------- | --------------------- | ----------------------------------------------------------------------------------------------- |
| 适配 Adapter         | 无内置 Strands 适配层 | `rllm/integrations/strands.py`：`RLLMModel` 适配 Strands `Model`，`StrandsAgent` 记录 rLLM 轨迹 |
| 工作流示例 Workflows | 无                    | `examples/strands/`：`strands_workflow.py`、`strands_math_workflow.py`、`research_workflow.py`  |
| 工具 Tools           | rLLM 自身工具体系     | Strands 示例中工具以函数对象注入；`RLLMModel.stream` 暂不处理 `tool_specs`                      |

## 工具/记忆 Tools and Memory

- rllm：
  - `rllm/tools` 提供工具基类与具体工具（如 code/web 等），用于 `ToolAgent`/特定环境。
- rllm-zero v0.2：
  - 新增 `rllm/tools/registry.py`（若存在）统一注册工具；
  - Strands 工作流中工具通过函数注入，未与 rLLM 工具声明（`tool_specs`）完全桥接；
  - 记忆（mem0/MEM1 等）由使用方在 Agent 层自行集成（示例层面）。

## 训练与配置 Training and Configuration

| 维度 Dimension         | rllm                                          | rllm-zero v0.2                                                                              |
| ---------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------- |
| 训练主线 Training path | `AgentExecutionEngine` 采样 → Trainer（verl） | 二选一：1) `AgentExecutionEngine` 标准采样；2) `AgentWorkflowEngine` 产生 veRL-ready 批数据 |
| Stepwise Advantage     | 需在上层配置/实现                             | `AgentWorkflowEngine` 的 veRL 转换内置支持                                                  |
| Compact Filtering      | 上层自定义                                    | `AgentWorkflowEngine` 的 veRL 转换内置支持                                                  |

## 示例与 API Examples and API

- rllm：
  - 多个任务型示例（math、code、browsergym、swe 等），偏向 Agent/Env 形态与训练脚本。
- rllm-zero v0.2：
  - `examples/strands/` 提供 Strands 工作流（研究、数学、并行执行）；
  - `rllm/api/api_server.py` 提供 API server 示例，内置默认模型、并行度与工作流配置；
  - 脚本中默认使用 OpenAI 兼容端点（如本地 `:30000/v1`）。

## 适用场景建议 When to use which

- 优先使用 rllm（Agent/Env 采样） Prefer rllm when:

  - 需要标准 RL 采样闭环、与现有 `BaseAgent`/`BaseEnv` 直接对接、或需要多返回模式（Token/Step/Conversation）。

- 优先使用 rllm-zero v0.2（Workflow 编排） Prefer rllm-zero v0.2 when:
  - 需要高层工作流编排（单/多 Agent）、Episode 存储、以及直接产出 veRL 训练所需张量的场景；
  - 需要复用 `RLLMModel` 对 Strands 进行推理适配并输出 rLLM 轨迹。

## 总结 Summary

- rllm-zero v0.2 在 rllm 的基础上引入 `Workflow` 抽象与 `AgentWorkflowEngine`，并提供 Strands 适配与示例，强化了高层编排、评测与训练数据转换能力；
- rllm 保持标准化的 Agent/Env 采样路径，更贴近传统 RL 训练流程；
- 两者可互补：在需要工作流与 Episode→veRL 转换时采用 rllm-zero v0.2；在需要传统 RL 采样与自定义 Agent/Env 时采用 rllm。
