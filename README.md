# Auto Research

面向 AI 论文复现的轻量级单 Agent 原型。当前输入由论文、作者代码仓库和待验证结论组成；Agent 在一次模型会话中完成论文与代码对齐、环境诊断、必要的兼容性修改、实验执行和复现报告生成。

当前版本用于验证“直接调用网络模型能够把论文复现推进到什么程度”，暂不包含 Planner、MAS、AgentBeats、隐藏评测器或 CORE-Bench Capsule 运行时。

## 目录结构

```text
auto-reproduction/
|-- auto-reproduction-agent/                # 单 Agent 实现、配置和测试
|   |-- reproducer/
|   |-- tests/
|   |-- pyproject.toml
|   `-- README.md
|-- tasks/                                  # 可提交的轻量任务定义
|   |-- culp/
|   |-- label_aware_gcn/
|   |-- ctgcn/
|   |-- multiagent_debate/
|   |-- catalog.json
|   `-- README.md
|-- resources/                              # 本地论文与仓库缓存，Git 忽略
`-- .gitignore
```

每个 `tasks/<task>/task.json` 保存：

- 论文下载地址和 SHA-256。
- 官方仓库地址和精确 commit。
- 需验证结论、建议实验和执行预算。

实际的 `paper.pdf` 与 `repository/` 在服务器运行准备命令后写入 `resources/<task>/`，不会上传 GitHub。

## 当前任务

| Task | 方向 | 定位 |
| --- | --- | --- |
| `culp` | 图半监督分类 | 最低成本的 CPU smoke test |
| `label_aware_gcn` | 图神经网络、轨迹预测 | 中等复杂度 |
| `ctgcn` | 时序图神经网络 | 较复杂的训练与评估任务 |
| `multiagent_debate` | LLM 多 Agent 推理 | 推荐的现代 API 实验；Agent 需自行获取 GSM8K |

前三项来自 CORE-Bench 公开训练元数据；Multiagent Debate 直接来自 ICML 2024 论文及作者官方仓库。详细来源见 `tasks/catalog.json` 和各任务的 `task.json`。

## 获取项目

项目仓库只保存 Agent 和任务清单，因此普通克隆即可：

```bash
git clone <your-repository-url>
cd auto-reproduction
```

首次创建并上传 GitHub 仓库时，在 `auto-reproduction/` 根目录执行：

```bash
git init -b main
git add .
git commit -m "Initial paper reproduction agent"
git remote add origin <your-repository-url>
git push -u origin main
```

## 安装

需要 Python 3.10 或更高版本：

```bash
cd auto-reproduction-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

PowerShell 激活命令为：

```powershell
.\.venv\Scripts\Activate.ps1
```

配置 OpenAI-compatible Chat Completions API：

```bash
export REPRO_API_BASE="https://api.openai.com/v1"
export REPRO_API_KEY="your-key"
export REPRO_MODEL="your-model-name"
```

不要把真实 API Key 写入仓库或任务文件。

## 运行

服务器首次克隆后，先下载任务声明的论文并克隆固定版本的作者仓库：

```bash
prepare-reproduction-task --task ../tasks/multiagent_debate/task.json
```

资源将保存到 `../resources/multiagent_debate/`。重复执行时会校验并复用已有资源。该准备器只获取输入中明确给出的论文和仓库；GSM8K 等实验依赖仍由复现 Agent 在运行中自行发现和获取。

然后进行不调用模型、不执行论文代码的输入检查：

```bash
python -m reproducer.cli \
  --task ../tasks/multiagent_debate/task.json \
  --output runs/multiagent-debate-check \
  --prepare-only
```

正式运行时移除 `--prepare-only`。每次运行会在输出目录保存隔离的代码副本、论文文本、工具调用轨迹、实验产物和最终 Markdown 报告。

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 当前边界

Agent 会在独立的运行目录中修改作者代码，但其命令执行目前仍继承宿主 Python 进程的权限。因此只应在一次性服务器或其他隔离环境中运行可信仓库。扩展到开放论文仓库之前，需要增加 Docker 或等价的网络与文件系统隔离。

Multiagent Debate 使用的原始模型 `gpt-3.5-turbo-0301` 已停用。该任务被定义为在同一当前可用模型上比较单 Agent 基线与 Debate 的“当代重验证”，不能表述为对原论文历史数值的精确复现。
