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
|   |-- h2o/
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
| `h2o` | LLM KV Cache 压缩 | 复现 Figure 4 左上角 XSUM/LLaMA-7B/ROUGE-2 完整曲线 |
| `streamingllm` | 流式 LLM / Attention Sink | 以 Pythia-2.8B 适配 Figure 3 的 20K-token 主实验协议 |

H2O 任务来自 NeurIPS 2023 论文及作者官方仓库。它需要本地运行 LLaMA-7B 的九组配置，属于高成本 GPU 实验，不是 smoke test。StreamingLLM 任务来自 ICLR 2024，比较固定 1024-token cache 下的 Window Attention 与 StreamingLLM；为控制单卡成本，它使用论文同一模型族的 Pythia-2.8B，而不是 Figure 3 中的 Pythia-12B，因此明确归类为主实验协议适配。详细范围见 `tasks/catalog.json` 和各任务目录。

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
prepare-reproduction-task --task ../tasks/h2o/task.json
```

运行 StreamingLLM 时将任务路径替换为
`../tasks/streamingllm/task.json`。准备器只下载任务声明的论文和固定提交仓库；
模型权重与 PG19 数据仍由 Agent 在隔离运行工作区中按任务协议获取。

资源将保存到 `../resources/h2o/`。重复执行时会校验并复用已有资源。该准备器只获取输入中明确给出的论文和仓库；模型权重和实验依赖仍由复现 Agent 在运行中自行发现和获取。

然后进行不调用模型、不执行论文代码的输入检查：

```bash
python -m reproducer.cli \
  --task ../tasks/h2o/task.json \
  --output runs/h2o-check \
  --prepare-only
```

若已配置视觉模型，可增加 `--prepare-visuals`，在正式 GPU 实验前检查 Figure 4 的定位、裁剪和数值提取。

正式运行时移除 `--prepare-only`。每次运行会在输出目录保存隔离的代码副本、论文文本、工具调用轨迹、实验产物和最终 Markdown 报告。

如果运行因步数上限以 `inconclusive` 结束，可直接复用原目录继续执行。例如原 H2O 运行停在第 55 步，再增加 80 步：

```bash
python -u -m reproducer.cli \
  --resume runs/h2o-main-20260826-112227 \
  --additional-steps 80 \
  2>&1 | tee -a runs/h2o-main-20260826-112227.console.log
```

续跑会保留原 workspace、已下载文件、安装结果和实验产物，并向原 `trace.jsonl` 追加第 56 步及后续记录。系统会从旧 trace 生成脱敏的 `workspace/resume_context.md` 交接摘要，而不是重新向模型发送完整历史。已经 `completed` 的运行不能续跑。

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 当前边界

Agent 会在独立的运行目录中修改作者代码，但其命令执行目前仍继承宿主 Python 进程的权限。因此只应在一次性服务器或其他隔离环境中运行可信仓库。扩展到开放论文仓库之前，需要增加 Docker 或等价的网络与文件系统隔离。

H2O 使用的原始 `huggyllama/llama-7b` 权重可能需要访问授权，且论文仓库依赖较旧的 Transformers 接口。替换模型只能作为协议适配实验，不能支持原论文特定结论。
