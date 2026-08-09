# 项目状态

最后更新：2026-08-08

## 当前阶段

8 月 15 日首发计划中的本地可执行部分已推进到：

- HotpotQA：检索核心完成；生成可信度问题已修正并重测。
- FinanceBench：数据适配、真实 PDF、页码验证、页内 Chunk、Global / Document-scoped 检索、Calculator 和 API 生成流水线完成。
- 求职交付：中文 README 与可提交结果摘要正在验收。
- 外部阻塞：真实 API 生成评测需要 Key；医疗 QLoRA 重训需要 24GB GPU；第二个公开仓库需要创建远端。

“代码存在”与“实验完成”严格分开：FinanceBench 当前没有答案准确率；医疗项目
当前只有历史 50 题结果，不能写成新 300 题实验。

## FinanceBench 数据验收

- 公开数据：150 题、84 份唯一文档。
- 有效语料：62 份 PDF、114 题、8,987 页、18,975 Chunks。
- 问题覆盖率：76.00%；文档覆盖率：73.81%。
- 已覆盖 gold evidence：143/143 在转换后的 1-based PDF 页精确匹配。
- 未转换的 0-based 页匹配：0/143，证明源字段需要 `+1`。
- 泄漏检查：索引 Chunk 中 question=0、answer=0、gold label=0。
- 5 份替代 PDF 因版式/分页不一致被隔离，没有进入索引。
- Dense 增量索引：复用 18,490 个向量，重新编码 485 个。

完整失败文档、URL 错误和逐题结果位于本地
`results/financebench_retrieval.json`；Git 只提交去逐题化摘要。

## FinanceBench 检索结果

114 题固定语料，Ranking depth 50，报告 Top-1/5/10。

### Global

| 系统 | Document Hit@10 | Page Hit@10 | Page Recall@10 | MRR | 主要 CPU 延迟 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.5614 | 0.0702 | 0.0702 | 0.0337 | 327.88 ms/q |
| Dense | **0.9737** | **0.3772** | **0.3377** | **0.1768** | Query encode 38.82 + search 81.40 ms/q |
| Hybrid | 0.9035 | 0.2456 | 0.2368 | 0.1129 | Fusion 0.26 ms/q |
| Reranker | 0.9561 | 0.3421 | 0.3158 | 0.1697 | 2,825.80 ms/q |

### Document-scoped

| 系统 | Page Hit@10 | Page Recall@10 | MRR | 主要 CPU 延迟 |
|---|---:|---:|---:|---:|
| BM25 | 0.2544 | 0.2544 | 0.1820 | 66.02 ms/q |
| Dense | **0.5965** | **0.5614** | **0.2925** | 4.49 ms/q（不含共享 Query encode） |
| Hybrid | 0.4298 | 0.4079 | 0.2175 | Fusion 0.19 ms/q |
| Reranker | 0.5439 | 0.5044 | 0.2637 | 2,748.37 ms/q |

Dense 是当前最强方案。Cross-Encoder 改善 Hybrid 但未超过 Dense，且延迟显著。
Global Dense Top-10 的 71 个失败中，68 个属于“正确财报已找到但证据页错误”，
3 个属于“正确财报未进入 Top-10”。

## FinanceBench 生成状态

已实现 OpenAI-compatible 结构化 Generator、安全 AST Calculator、模型引用校验、
断点续跑、token/延迟/模型元数据记录。最终 dry-run 结果：

- Document-scoped Dense Top-5。
- 114 题。
- 100% 引用页来自真实 PDF。
- 平均 Prompt 9,074.18 字符。
- 未发送 API 请求，未生成答案指标。

## HotpotQA 结果

检索设置：100 条 validation，20 Dev / 80 Test，Seed 42，Split SHA-256：
`08c3e1aa1a8fe2844f899684db7e1cfd1d2f0915e31edafe1b17c8a795891660`。

Cross-Encoder Test：Hit@1 0.9000、Recall@10 0.9094、Complete@10 0.8000、
MRR 0.9378；CPU 增量延迟 427.80 ms/q。

可信度修复：删除 nationality 特例、Title Promotion 和强制引用注入。修复后
FLAN-T5-small 80 题生成结果：

- EM 0.2625；Token F1 0.3518。
- Has Citation 0.1500；Citation Validity 0.1500。
- Citation Precision 0.1500；Gold-fact Citation Recall 0.0667。
- 生成延迟 518.81 ms/q。

旧的 100% 引用报告已保存为
`results/hotpotqa_generation_sample100_test80_pre_citation_fix_invalid.json`，明确视为
无效历史结果，不用于 README、简历或 Notion 结论。

## 2026-08-07 实际执行记录

```powershell
& '.\.conda\python.exe' '.\scripts\prepare_financebench.py' ...
& '.\.conda\python.exe' '.\scripts\validate_financebench_pages.py'
& '.\.conda\python.exe' '.\scripts\run_financebench_retrieval_experiment.py'
& '.\.conda\python.exe' '.\scripts\run_hotpotqa_generation_evaluation.py'
& '.\.conda\python.exe' '.\scripts\run_financebench_generation_evaluation.py' --dry-run
& '.\.conda\python.exe' '.\scripts\publish_experiment_summaries.py'
```

最终验收：59 项测试全部通过（3.85 秒），`compileall` 通过，公开摘要可重复
生成。Demo 使用线程安全单例与启动预热；冷启动 49.6 秒后 ready，
实际样例回答 `Scotch Collie`，检索 40.16 ms、重排 422.23 ms、生成
430.95 ms、总计 893.34 ms。

Notion 的 EvalRAG 首页、Roadmap、Atomic Actions 和 Experiments & Results 已于
2026-08-08 按实际结果更新；原有子页保留，旧的 100% 引用结论已移除。

## 未完成与风险

- 22 份财报未通过下载或页码一致性校验，不能声称 150/150。
- FinanceBench 真实答案、数值准确率、成本尚未测量。
- 通用 MiniLM 和 MS MARCO Cross-Encoder 存在金融域偏移。
- PDF 表格目前按文本抽取，复杂表格结构可能丢失。
- HotpotQA 本地生成器引用能力弱；引用有效不等于 claim-level faithfulness。
- Docker、公网页面、鉴权、日志和监控按首发计划暂缓。
