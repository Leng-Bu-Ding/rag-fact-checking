# 项目状态

最后更新：2026-08-14

## 当前阶段

8 月 15 日首发计划中的本地可执行部分已推进到：

- HotpotQA：检索核心完成；生成可信度问题已修正并重测。
- FinanceBench：数据适配、真实 PDF、页码验证、页内 Chunk、Global / Document-scoped 检索、Calculator、114题 API 生成和答案 Judge 完成。
- 求职交付：RAG PR #2 已通过 Squash merge 合入 `main`，114题生成与 Judge
  成果已正式发布。RAG 与医疗两个公开仓库均已建立。
- 文档治理：建立“Git 仓库保存可验证事实、Notion 保存个人认知、本机保存私有/大型产物”的三层体系。
- 外部阻塞：医疗 QLoRA 重训需要云端 CUDA GPU，优先使用24GB级显存并先做显存 smoke test。

“代码存在”与“实验完成”严格分开：FinanceBench 已有真实结果；医疗项目当前
只有历史50题结果，不能写成新300题实验。

## 发布与事实来源

- RAG 仓库：https://github.com/Leng-Bu-Ding/rag-fact-checking
- 医疗仓库：https://github.com/Leng-Bu-Ding/medical-llm-qlora
- FinanceBench 114题生成发布提交：`d9e51fd`（PR #2 Squash merge）。
- 可公开指标以 `results/public/*.json` 为底层事实来源。
- 参数以实际运行脚本读取的 `configs/*.yaml` 为准；Notion 只解释结论与决策。
- API Key、原始数据、PDF、模型权重、索引和完整逐题报告不进入 Git。

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

## FinanceBench 生成结果

已实现 OpenAI-compatible 结构化 Generator、安全 AST Calculator、模型引用校验、
断点续跑、Token/延迟/模型元数据记录，并完成 Document-scoped Dense Top-5 的
114题真实 API 生成。85题正常生成，29题生成流程失败；自动 Judge 给出43题正确、
20题错误、22题拒答和29题生成失败。详细指标、资源消耗和局限见本文后面的
“2026-08-13 FinanceBench 生成全量验收”，公开事实源为
`results/public/financebench_generation_summary.json`。

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

Notion 已按三层文档体系重构：新增环境资源、数据产物、代码配置页面，重写项目
定义、请求生命周期、当前行动和面试掌握页面；原有 Draft 标为历史。旧的 100%
引用结论已移除。

## 2026-08-10 文档治理验收

- 新增 `docs/architecture.md`、`docs/environment.md`、`docs/data-and-artifacts.md`、
  `docs/code-and-configuration.md`、`docs/experiments.md`、`docs/runbook.md`。
- README 增加文档导航；本机绝对模型缓存路径和真实密钥未写入 Git 文档。
- Notion 新增 Environment & Resources、Data & Artifacts、Code & Configuration；
  更新 Project Definition、Architecture、Roadmap、Current Actions、Decisions 与
  Interview & Mastery。
- `git diff --check` 通过，README 的9个本地文件链接存在。
- 沙箱内 pytest 因临时目录访问限制报 `WinError 5`；在沙箱外使用新的
  `--basetemp` 重跑，59项测试全部通过（4.62秒）。

## 2026-08-12 百炼 API 兼容性修复

- FinanceBench 单题真实调用已到达百炼，但首次返回 HTTP 400；失败发生在模型
  生成前，尚未形成有效预测记录。
- 根据百炼混合思考模型的接口约束，在结构化 JSON 输出请求中显式设置
  `enable_thinking: false`，避免 Qwen3.7 Plus 默认思考模式与 JSON Mode 冲突。
- HTTP 失败现在会报告百炼返回的状态码、错误码、消息和 request ID，但不会输出
  Authorization Header 或 API Key。
- 进一步发现 Python `requests` 默认继承 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`
  等环境代理；FinanceBench 配置现在显式设置 `trust_env_proxy: false`，百炼请求改为
  直连，避免本地代理或代理网关把请求改写成无错误详情的 HTTP 400。
- 百炼生成器定向测试通过：7 passed；完整测试在沙箱外通过：63 passed in 3.66s；
  `compileall` 与 `git diff --check` 通过。
- 2026-08-13 复测后纯文本 HTTP 400 仍存在。使用不含 API Key 的直连最小请求验证：
  业务空间 `/compatible-mode/v1/chat/completions` 正确返回百炼标准 JSON 401，证明
  DNS、TLS、业务空间域名和接口路径可达；问题进一步缩小到凭据内容或请求字段。
- 新增 `--probe-api` 安全分级诊断，依次检查最小 Chat、关闭思考和 JSON Mode；
  不打印 API Key、不读取或写入预测文件，失败后停止。定向测试 9 passed；完整测试
  在沙箱外 65 passed in 10.25s；`compileall` 与 `git diff --check` 通过。
- 下一步先在保存环境变量的同一终端运行 `--probe-api`；依据明确失败阶段修复后，
  再运行 `--limit 1`，核验预测、引用、Token和指标，然后决定是否运行114题。

## 2026-08-13 百炼凭据根因确认

- `--probe-api` 在 `minimal_chat` 阶段即失败，排除 FinanceBench Prompt、JSON Mode、
  `enable_thinking`、Calculator 和长上下文。
- 使用不带凭据的直连请求得到标准 JSON 401，证明业务空间域名、TLS 和 Chat
  Completions 路径可用。
- 经用户明确授权，从 PowerShell 历史中临时恢复最近一次 `RAG_API_KEY`，绕过环境
  代理后请求百炼；服务端明确返回 `HTTP 401`、`invalid_api_key`。密钥值未打印、
  未写入项目文件，诊断进程结束后已移除。
- 该历史值长度为39，既不以普通百炼 Key 的 `sk-` 开头，也不以 Token Plan Key
  的 `sk-sp-` 开头，格式不符合百炼 API Key；当前根因是错误凭据，而非项目代码。
- 已建立被 `.gitignore` 排除的本机 `.env.local` 模板，供新 Key 安全落地；文件不会
  被 Git 跟踪。下一步是在百炼控制台创建/重置有效 Key，然后自动重跑分级探测和
  FinanceBench 单题评测。

## 2026-08-13 百炼连通与 FinanceBench 单题验收

- 用户保存的新式普通百炼 Key 已通过真实接口验证；它以 `sk-` 开头、包含点号，
  三层探测 `minimal_chat`、`non_thinking_chat`、`json_mode` 全部成功。
- 脚本现在自动读取被 Git 忽略的 `.env.local`，且不覆盖终端中显式设置的环境变量；
  以后无需每次手动设置 Base URL、模型名和 Key。
- 脱敏规则已覆盖带点号的新式 Key；Key 未进入 Git、日志、预测文件或状态文档。
- FinanceBench 第1题真实端到端运行成功：Document-scoped Dense Top-5，引用
  Precision=1.0、Recall=1.0，Prompt 5,966 tokens、Completion 193 tokens，生成
  延迟6.120317秒。
- 该题检索到了正确证据页，但模型按传统口径计算 `7,453-5,175=2,278`；数据集
  标准口径排除现金和短期借款，计算 `1,721+2,904+1,157-1,804-3,147=831`，
  因而 EM=0、Numeric Match=0。不能把这项错误归因于检索，应作为财务口径选择
  错误处理，且不得针对单题硬编码。
- 自动加载探测通过；完整测试在沙箱外 66 passed in 5.06s，`compileall` 与
  `git diff --check` 通过。

## 2026-08-13 FinanceBench 生成全量验收

- 生成流程升级为 `financebench-v5`：按问题触发财务定义，模型返回 metric
  definition、selected items 与 expression；Python 校验经营性 working capital
  行项目覆盖、跨期变化百分比和计划完整性，再由安全 Calculator 执行算式。
- 固定4题 Dev 与4题 Holdout；Prompt 只根据 Dev 修改，Holdout 运行一次后冻结。
  working capital Dev 从错误的2,278修正为正确的831；Dev 数值正确3/4，剩余1题
  因标准证据页未进入 Top-5 而拒答。
- 114题全量：85题正常生成、29题生成流程失败、22题明确拒答；46题的全部标准
  证据页进入 Dense Top-5。
- 确定性总体指标：EM 0.0614、Numeric Match 0.1842、Citation Precision 0.2558、
  Citation Recall 0.2939；EM 对自由文本和近似数值等价答案明显过严。
- `financebench-answer-judge-v1` 对63个非拒答答案评测：43正确、20错误；加上
  22拒答和29生成失败后，端到端总体正确率43/114=0.3772；实际给出可评判答案
  时正确率43/63=0.6825；全部 gold 页进入 Top-5 的46题总体正确率0.5870。
- Judge 错误类型：wrong value 11、incomplete 3、unsupported claim 3、wrong
  direction 3。Judge 与生成器使用同一模型家族，只能视为自动语义评测，不是独立
  人工审计。
- 生成使用310,021输入/20,577输出 Token，累计模型延迟650.19秒；Judge 使用
  17,804输入/5,040输出 Token，延迟153.45秒。使用免费额度，未读取账单，不能把
  Token 推算值写成实际成本。
- 完整逐题结果只保存在本机 `results/`；Git 公开摘要为
  `results/public/financebench_generation_summary.json`，不包含逐题数据或密钥。
- 最终完整回归75 passed in 4.00s；`compileall`、公开摘要生成、密钥扫描、README
  链接和 `git diff --check` 均通过。

## 未完成与风险

- 22 份财报未通过下载或页码一致性校验，不能声称 150/150。
- FinanceBench 自动答案质量、Token和延迟已测量；实际账单成本及人工正确率未测量。
- 通用 MiniLM 和 MS MARCO Cross-Encoder 存在金融域偏移。
- PDF 表格目前按文本抽取，复杂表格结构可能丢失。
- HotpotQA 本地生成器引用能力弱；引用有效不等于 claim-level faithfulness。
- Docker、公网页面、鉴权、日志和监控按首发计划暂缓。
