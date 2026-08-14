# 系统架构与请求生命周期

最后核验：2026-08-10

## 系统目标

EvalRAG 将“检索是否正确”“答案是否正确”“引用是否支持答案”分开处理和评测。
HotpotQA 用于验证通用多跳检索，FinanceBench 用于验证真实财报 PDF、页码证据和
数值问答。两条实验线复用检索、重排、生成和评测接口。

## 离线数据链路

```text
远端数据集 / PDF
→ Adapter 标准化
→ 清洗与按页解析
→ Chunk + Provenance
→ BM25 语料 / Dense Embedding
→ FAISS Index + Manifest
→ 固定实验划分与结果文件
```

HotpotQA Chunk 保留 `sample_id / title / sentence_ids`；FinanceBench Chunk 保留
`financebench_id / document_id / company / year / document_type / page_number`。
这些字段使检索结果能够回到原始证据。

## 在线问答链路

```text
Question
→ BM25 Top-N
→ Dense FAISS Top-N
→ Hybrid RRF
→ Cross-Encoder Reranking
→ Evidence Top-K
→ Local FLAN-T5 或 OpenAI-compatible Generator
→ Citation Validation
→ Answer + Evidence + Timing
```

### HotpotQA Demo

1. `app/api.py` 在启动阶段创建线程安全的 `RAGService` 单例。
2. `src/services/rag.py` 加载 Dense 索引、BM25、Embedding、Reranker 和生成器。
3. BM25 与 Dense 各自返回候选结果。
4. RRF 融合候选，Cross-Encoder 对候选重新打分。
5. `google/flan-t5-small` 在本机 CPU 上根据 Top-K 证据生成答案。
6. 系统只保留模型实际生成且编号合法的引用，不自动补引用。
7. API 返回证据、分数、检索/重排/生成分阶段耗时。

### FinanceBench 评测

1. 在 62 份页码对齐的真实 PDF 中检索114道有效覆盖问题。
2. Global Retrieval 不预先知道相关财报；Document-scoped Retrieval 已知相关财报。
3. 当前生成配置使用 Document-scoped Dense Top-5。
4. 本地代码选择财务定义、校验行项目计划并用安全 Calculator 执行算式；真实答案由 OpenAI-compatible API 生成。
5. 固定 Dev 用于开发 Prompt，Holdout 只运行一次；全量结果按 Prompt 版本独立保存。
6. 结构化 Judge 比较问题、参考答案与候选答案，并与确定性指标分开报告。
5. Calculator 只执行受限算术表达式，不执行任意 Python。
6. 逐题保存答案、引用、模型元数据、Token 和分阶段延迟，并支持断点续跑。

## 运行边界

| 模块 | 运行位置 | 是否需要网络/Key |
|---|---|---|
| BM25、FAISS Search、RRF | 本机 CPU | 已有数据后不需要 Key |
| MiniLM Embedding、Cross-Encoder | 本机 CPU | 首次下载模型需要网络，不需要 API Key |
| HotpotQA FLAN-T5 | 本机 CPU | 首次下载模型需要网络，不需要 API Key |
| FinanceBench 强模型生成 | 云端模型服务 | 需要网络、Base URL、Model、API Key |
| 医疗 QLoRA | 独立仓库与云端 GPU | 需要 CUDA GPU，模型下载可能需要 Token/License |

## 当前主要瓶颈

- HotpotQA：检索较强，但 FLAN-T5-small 的答案与引用能力较弱。
- FinanceBench：文档定位较强，页级证据定位明显更难。
- 通用 MS MARCO Reranker 存在金融域偏移，且 CPU 延迟高。
- FinanceBench 已完成114题真实生成；同模型 Judge 不是独立人工审计，Token 也不等于实际账单金额。
