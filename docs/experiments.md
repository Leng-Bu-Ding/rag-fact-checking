# 实验登记与事实来源

最后核验：2026-08-10

## 状态定义

| 状态 | 精确定义 |
|---|---|
| Planned | 只有计划 |
| Implemented | 代码存在，但未形成真实结果 |
| Dry-run | 输入、Prompt和输出路径通过检查，未调用外部模型 |
| Measured | 在固定数据和配置上实际运行并保存结果 |
| Published | 去逐题化摘要进入Git，README/Notion数字一致 |

“Implemented”不能写成“Measured”，“Dry-run”不能产生答案准确率。

## 实验登记

| 编号 | 实验 | 状态 | 公开事实来源 |
|---|---|---|---|
| H01 | HotpotQA BM25/Dense/Hybrid/Reranker | Published | `results/public/hotpotqa_retrieval_summary.json` |
| H02 | HotpotQA诚实生成重测 | Published | `results/public/hotpotqa_generation_summary.json` |
| F01 | FinanceBench数据与页码审计 | Published | `results/public/financebench_retrieval_summary.json` |
| F02 | FinanceBench Global Retrieval | Published | 同上 |
| F03 | FinanceBench Document-scoped Retrieval | Published | 同上 |
| F04 | FinanceBench API Generation | Published | `results/public/financebench_generation_summary.json` |
| M01 | 医疗8B QLoRA重训与300+50评测 | Planned/Blocked | 独立仓库；等待GPU |

## 实验必须记录

- 研究问题和对照组。
- 数据集、样本数、Dev/Test划分、Seed和数据版本。
- 模型名、revision、关键参数和设备。
- 输入报告或索引的SHA。
- 检索、重排、生成的分阶段延迟。
- 指标、逐题失败类型和局限。
- 实际命令、输出路径和是否产生外部费用。

## 发布规则

1. 完整逐题记录保留在本地 `results/`。
2. `scripts/publish_experiment_summaries.py` 生成可公开摘要。
3. `results/public/*.json` 是公开数字的底层事实来源。
4. README只摘录最重要指标。
5. Notion解释实验为什么这样设计、为什么得到该结果。
6. 简历只使用已Published且能指向结果文件的数字。

## 无效历史结果

强制补引用产生的旧HotpotQA 100%引用报告已撤回，只作为失败教训保留在本地，
不得用于README、Notion结论、简历或面试陈述。
