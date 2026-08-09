# 待办事项

最后更新：2026-08-08

## 下一里程碑：补齐外部资源实验并完成 8 月 15 日首发

### P0 · 需要用户提供资源

- [ ] 提供 OpenAI-compatible API 的 Base URL、Key 和 Model，运行 FinanceBench
  114 题端到端答案评测；报告答案 EM/F1、数值准确率、引用、检索/重排/生成
  延迟、token 和实际成本。
- [ ] 提供 24GB GPU 平台并确认模型 License，运行独立
  `medical-llm-qlora` 仓库的 1 epoch QLoRA 重训、固定 300 条 Base/FT 对照和
  50 条安全案例。
- [ ] 安装并登录 GitHub CLI，或手动创建空的 `medical-llm-qlora` 远端仓库；
  在公开前核对两个仓库不包含密钥、数据、PDF、模型权重或本机路径。

### P0 · 本仓库最终验收

- [x] 运行完整 pytest（59 passed）、compileall、FinanceBench dry-run 和 FastAPI Demo smoke。
- [ ] 核对 `results/public/` 与 README、Notion、简历中的每个数字一致。
- [x] 将 FinanceBench 数据、检索、错误分析和真实阻塞同步到 Notion。
- [ ] API 结果完成后再修改简历，不预写答案准确率或 150/150 覆盖。

### P1 · 8 月 15 日后增强

- [ ] 为 FinanceBench 比较页内 Chunk 大小、表格感知切块和金融域 Reranker。
- [ ] 增加 Supported/Unsupported 拒答集与 claim-level faithfulness。
- [ ] HotpotQA 扩容到 1,000 或完整 Validation，并记录时间/内存/索引曲线。
- [ ] 增加 Docker、CI、结构化日志、鉴权和公网 Demo。
- [ ] 每周完成一次 60 分钟项目深挖模拟面试。

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

目前 RAG 检索部分已达到可展示状态，但整套 8 月 15 日方案尚未完成。只有在
API 真实答案实验、医疗 GPU 重训、两个公开仓库和最终 smoke 全部通过后，才能
把首发计划标记为完成。
