# 环境、模型与资源要求

最后核验：2026-08-10

## 软件环境

| 项目 | 当前要求 |
|---|---|
| 操作系统 | Windows + PowerShell；代码保持 Python 跨平台路径语义 |
| Python | 3.11.x |
| 环境管理 | Conda，仓库内前缀环境 `.conda/` |
| 解释器 | `./.conda/python.exe`，不得假设全局 PATH 有 Python |
| 运行依赖 | `requirements.txt` |
| 测试依赖 | `requirements-dev.txt` |
| 版本管理 | Git；`main` 为发布主线 |

VS Code 只是编辑、运行和调试入口；真正执行代码的是项目 Conda 环境中的 Python。

## 当前 RAG 模型

| 模型 | 用途 | 执行位置 | 本机已测缓存目录逻辑大小 |
|---|---|---|---:|
| `sentence-transformers/all-MiniLM-L6-v2` | Dense Embedding | CPU | 约87 MiB |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 候选重排 | CPU | 约88 MiB |
| `google/flan-t5-small` | HotpotQA 答案生成 | CPU | 约590 MiB |

模型由 Hugging Face Hub 下载并缓存在用户缓存目录。默认由 Hugging Face 决定
具体路径；如需迁移到其他磁盘，只在本机设置 `HF_HOME`，不得把绝对路径写入
Git 跟踪配置。

`configs/default.yaml` 中的 Qwen/BGE 是未接入当前 Demo 的历史规划字段，不能据此
声称本机已经下载或运行 Qwen。当前 Demo 的实际模型以 `src/services/rag.py`、
`configs/dense.yaml`、`configs/hotpotqa_retrieval.yaml` 和运行结果元数据为准。

## 外部服务

FinanceBench 真实生成使用 OpenAI-compatible HTTP 接口：

| 环境变量 | 含义 | 是否敏感 |
|---|---|---|
| `RAG_API_BASE_URL` | 服务端点 | 否 |
| `RAG_API_MODEL` | 模型标识 | 否 |
| `RAG_API_KEY` | 身份、额度和计费凭证 | 是 |

Key 只能放在当前终端环境变量或未提交的 `.env` 中，不能写入代码、配置、日志、
Notion、截图或 GitHub。当前 dry-run 不发送 API 请求，也不产生模型费用。

## 计算资源

- 本地 RAG 检索和 HotpotQA Demo 使用 CPU，可在当前电脑运行。
- FinanceBench API 生成在服务商服务器运行，本机主要负责构造请求和保存结果。
- 医疗 Llama 3 8B QLoRA 位于独立仓库。本机 GTX 1050 4GB 不作为训练设备。
- 医疗训练优先租用 A10 等 24GB 级显存；若使用 T4 16GB，必须先运行显存 smoke
  test，再决定是否调整 batch、sequence length 或 checkpointing。

24GB 是降低训练中断风险的当前目标，不是没有实测依据的“绝对最低显存”。

## 存储与费用影响

| 产物 | 典型位置 | Git | 费用/影响 |
|---|---|---|---|
| Conda 环境 | `.conda/` | 忽略 | 占磁盘，可由依赖重建 |
| 数据/PDF | `data/raw/` | 忽略 | 占磁盘，可部分重新下载 |
| Chunk/索引 | `data/processed/`、`data/index/` | 忽略 | 可重新生成但耗时 |
| 本地模型 | Hugging Face cache | 忽略 | 首次下载耗网络与磁盘 |
| 完整结果 | `results/` | 忽略 | 保留实验审计；覆盖前记录配置 |
| 公开摘要 | `results/public/` | 跟踪 | 小型、可核验 |
| API 生成 | 云端 + 本地结果 | 摘要可跟踪 | 按服务商 Token 计费 |
| 医疗训练 | 云端 GPU | 权重不进 Git | 按 GPU 使用时长计费 |

## 密钥检查

提交前至少运行：

```powershell
git status --short
git diff --cached
rg -n "API_KEY|Bearer |sk-" . -g '!data/**' -g '!results/**' -g '!.git/**'
```

搜索只能辅助检查，不能替代人工确认。
