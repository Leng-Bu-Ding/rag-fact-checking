# 数据与产物生命周期

最后核验：2026-08-10

## 两条数据线

| 数据集 | 作用 | 远端来源 | 当前实验范围 |
|---|---|---|---|
| HotpotQA | 多跳检索算法验证 | `hotpotqa/hotpot_qa` | distractor validation 前100条，20 Dev / 80 Test |
| FinanceBench | 真实金融 PDF 业务评测 | `PatronusAI/financebench` | 150题清单；62份有效PDF覆盖114题 |

HotpotQA 的 `supporting_facts` 和 FinanceBench 的 gold evidence 只用于评测或显式
调试，不能进入索引文本、Embedding 或 Reranker 输入。

## 目录生命周期

```text
data/raw/        原始数据集缓存、问题清单、PDF、下载 Manifest
      ↓
data/processed/  标准化样本和保留来源信息的 Chunk
      ↓
data/index/      Dense 向量、FAISS 索引和 Manifest
      ↓
results/         完整逐题实验、日志和失败记录
      ↓
results/public/  去逐题化、可提交、可核验的真实摘要
```

前三个 `data/` 目录和完整 `results/` 都被 Git 忽略。`results/public/` 使用
`.gitignore` 例外规则进入版本管理。

## FinanceBench 页码规则

- `evidence_page_num` 已验证为0-based。
- 用户看到的真实 PDF 页码为1-based，因此索引元数据执行 `+1`。
- Chunk 禁止跨页，引用才能稳定对应单一 PDF 页。
- 替代 PDF 必须通过 gold 文本与目标页精确对齐，分页不一致文件被隔离。
- Gold evidence 文本不能被当作“财报语料”索引，否则会造成评测泄漏。

## 当前可核验规模

- FinanceBench 公开：150题、84份唯一文档、189条 gold evidence。
- 严格有效语料：62份PDF、114题、8,987页、18,975 Chunks。
- 已覆盖 evidence：143/143 在转换后的1-based页匹配。
- 索引泄漏检查：question=0、answer=0、gold label=0。

## 删除与恢复影响

| 内容 | 能否恢复 | 代价 |
|---|---|---|
| 原始 HotpotQA 缓存 | 通常可重新下载 | 网络与下载时间 |
| FinanceBench PDF | 部分链接可能失效 | 不保证完全恢复，禁止随意删除 |
| Processed Chunk | 可以 | 需要重新解析和切块 |
| Dense Index | 可以 | 需要重新编码；耗 CPU 时间 |
| 完整逐题结果 | 取决于配置和外部模型 | API结果可能再次产生费用，应保留 |
| `results/public/` | 可由完整结果发布 | 应由 Git 保留版本 |

## 事实来源

- 数据路径和参数：`configs/*.yaml`。
- 数据处理实现：`src/data/`、`src/pipelines/`、`scripts/prepare_financebench.py`。
- 公开规模与指标：`results/public/*.json`。
- 当前阶段：`PROJECT_STATUS.md`。
