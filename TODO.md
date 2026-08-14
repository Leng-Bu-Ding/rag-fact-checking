# 待办事项

最后更新：2026-08-14

## 下一里程碑：补齐外部资源实验并完成 8 月 15 日首发

### P0 · 需要用户提供资源

- [x] 创建有效的按量付费 API Key，写入被 Git 忽略的 `.env.local`；三层 API
  探测和 FinanceBench 单题端到端评测已通过。
- [x] 完成可泛化财务定义、结构化行项目计划、安全 Calculator、固定 Dev/Holdout
  和114题真实生成；发布确定性指标、同模型 Judge、引用、延迟和 Token。实际成本
  未从账单读取，不做推测。
- [ ] 提供云端 CUDA GPU 并确认模型 License；优先选择 A10 等 24GB 级显存，
  若使用 T4 16GB 则先运行显存 smoke test。随后运行独立
  `medical-llm-qlora` 仓库的 1 epoch QLoRA 重训、固定 300 条 Base/FT 对照和
  50 条安全案例。

### P0 · 本仓库最终验收

- [x] 运行完整 pytest（75 passed）、compileall、FinanceBench dry-run、API 生成和 FastAPI Demo smoke。
- [ ] 核对 `results/public/` 与 README、Notion、简历中的每个数字一致。
- [x] 将 FinanceBench 数据、检索、错误分析和真实阻塞同步到 Notion。
- [x] 发布 RAG 与 `medical-llm-qlora` 两个公开仓库，并完成隐私与文件范围检查。
- [x] 建立公开 `docs/` 文档体系，分离架构、环境、数据、配置、实验和运行手册。
- [ ] 基于已发布的114题真实结果修改简历；明确自动 Judge 局限，不声称150/150覆盖。

### P1 · 8 月 15 日后增强

- [ ] 为 FinanceBench 比较页内 Chunk 大小、表格感知切块和金融域 Reranker。
- [ ] 增加 Supported/Unsupported 拒答集与 claim-level faithfulness。
- [ ] HotpotQA 扩容到 1,000 或完整 Validation，并记录时间/内存/索引曲线。
- [ ] 增加 Docker、CI、结构化日志、鉴权和公网 Demo。
- [ ] 统一 Demo 生成模型的配置入口，移除 `configs/default.yaml` 中未生效的历史模型字段。

## 已完成：FinanceBench Retrieval

- [x] 150 题 Adapter 与 84 文档 Manifest。
- [x] 并发 PDF 下载、SHA-256、页数和失败原因记录。
- [x] PyMuPDF 按页解析、1-based 页码和禁止跨页 Chunk。
- [x] gold evidence 不进入索引，问题/答案/gold 泄漏均为 0。
- [x] 替代 PDF 精确页码验证和不一致隔离。
- [x] BM25、Dense、Hybrid、Cross-Encoder 的 Global / Document-scoped 评测。
- [x] Document Hit、Page Hit/Recall、MRR、NDCG、分阶段延迟和逐题结果。
- [x] 按“正确页 / 正确文档错误页 / 错误文档”聚合失败类型。
- [x] Dense 增量索引复用。
- [x] 结构化 API Generator、Calculator、引用验证和 dry-run。

## 已完成：HotpotQA 可信度修复

- [x] 删除 nationality 硬编码和 Title Promotion。
- [x] 不再给无引用答案强制添加 `[1]`。
- [x] 网页正确显示 Cross-Encoder logit、重排耗时和引用策略。
- [x] 重跑 80 题生成评测并撤回旧的虚高引用结论。

## 首发验收边界

RAG 仓库的本地首发候选已经完成真实答案实验和最终 smoke，当前剩余工作是将
本次变更通过 PR 发布、统一简历/Notion 数字并完成人工抽样复核。整个双项目组合
仍未完成，因为独立医疗仓库的 GPU 重训、300题对照和50条安全案例尚未运行。
