# 代码与配置地图

最后核验：2026-08-10

## 目录职责

| 目录 | 职责 | 是否直接运行 |
|---|---|---|
| `src/data/` | 数据适配、JSONL、PDF与Chunk | 通常由脚本调用 |
| `src/retrieval/` | BM25、Dense、RRF、Reranker | 通常由Pipeline调用 |
| `src/generation/` | 本地生成、API生成、Calculator | 通常由服务/脚本调用 |
| `src/evaluation/` | 检索、答案、引用指标 | 通常由实验脚本调用 |
| `src/pipelines/` | 组合模块形成离线流程 | 由脚本调用 |
| `src/services/` | 在线RAG服务 | 由FastAPI调用 |
| `scripts/` | 可执行入口 | 是 |
| `app/` | FastAPI与静态页面 | 由启动脚本调用 |
| `configs/` | 参数、路径、模型名 | 被代码读取 |
| `tests/` | 自动验证行为 | 由pytest运行 |

## 从命令到产物

| 任务 | 入口 | 主要配置 | 主要输出 |
|---|---|---|---|
| 查看HotpotQA样本 | `scripts/run_hotpotqa_sample.py` | `configs/default.yaml`的数据段 | 终端预览 |
| 构建HotpotQA Chunk | `scripts/build_hotpotqa_chunks.py` | `configs/default.yaml` | `data/processed/*.jsonl` |
| Dense索引 | `scripts/run_dense.py` | `configs/dense.yaml` | `data/index/` |
| HotpotQA检索实验 | `scripts/run_hotpotqa_retrieval_experiment.py` | `configs/hotpotqa_retrieval.yaml` | `results/*.json` |
| HotpotQA生成实验 | `scripts/run_hotpotqa_generation_evaluation.py` | 检索报告 + 生成器默认值 | `results/*.json` |
| FinanceBench准备 | `scripts/prepare_financebench.py` | `configs/financebench.yaml` | 数据、PDF、Chunk |
| FinanceBench检索 | `scripts/run_financebench_retrieval_experiment.py` | `configs/financebench.yaml` | 检索报告 |
| FinanceBench生成 | `scripts/run_financebench_generation_evaluation.py` | 配置 + 环境变量 | 预测与摘要 |
| 本地Demo | `scripts/start_demo.py` | Dense、HotpotQA、Demo配置 | HTTP服务 |

## 配置优先级

1. 命令行显式参数覆盖脚本默认值。
2. 脚本读取对应的 `configs/*.yaml`。
3. 密钥只从环境变量读取，不得写进YAML。
4. 实验报告必须保存实际模型元数据和关键配置，不能只引用“默认值”。

## 当前配置债务

`configs/default.yaml` 同时含有数据、预处理和历史的BGE/Qwen规划字段，但当前
Demo不读取其中的BGE/Qwen：

- Dense实际使用 `configs/dense.yaml` 中的 `all-MiniLM-L6-v2`。
- Reranker实际使用 `configs/hotpotqa_retrieval.yaml` 中的MS MARCO模型。
- Demo生成器当前由 `LocalGroundedGenerator` 默认加载 `google/flan-t5-small`。

因此不能看到YAML中的模型名就断言系统实际使用该模型。后续应统一生成器配置
入口并删除未生效字段；在完成行为测试前，不把配置清理与实验改动混在一起。

## 阅读代码的推荐顺序

1. `scripts/run_hotpotqa_retrieval_experiment.py`
2. `src/data/load_hotpotqa.py`
3. `src/data/chunking.py`
4. `src/retrieval/bm25.py`
5. `src/retrieval/dense.py`
6. `src/retrieval/hybrid.py`
7. `src/retrieval/reranker.py`
8. `src/evaluation/retrieval.py`
9. `src/generation/grounded.py`
10. `src/services/rag.py`
11. `app/api.py`

阅读每个模块时固定回答：输入、输出、配置、依赖、产物、指标和失败方式。
