# 公开实验结果

这里仅保存可随 Git 提交的小型、去逐题化实验摘要：

- `hotpotqa_retrieval_summary.json`：固定 20 Dev / 80 Test 的检索指标和延迟。
- `hotpotqa_generation_summary.json`：删除强制引用后的真实生成指标。
- `financebench_retrieval_summary.json`：真实 PDF 覆盖、两种检索范围、失败类型和延迟。
- `financebench_generation_summary.json`：114题 API 生成、确定性指标、Judge verdict、Token、延迟和局限。

完整逐题报告、PDF、数据、索引和模型缓存保留在本机并由 `.gitignore` 排除。
摘要由以下命令从本地完整报告生成：

```powershell
& '.\.conda\python.exe' '.\scripts\publish_experiment_summaries.py'
```

FinanceBench 逐题预测与 Judge 原因保留在本机 `results/`；Git 只提交去逐题化摘要。
