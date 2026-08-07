# AGENTS.md

修改项目前，必须阅读 `PROJECT_STATUS.md`、`TODO.md` 和本文件。如果存在
`AGENTS.local.md`，还要读取其中记录的机器专属路径。

## 项目使命

构建一个具备生产意识、能够在面试中清楚解释的 RAG 系统。优先保证可测量的
检索质量、有依据的答案、可追踪的证据和可复现实验，而不是盲目增加功能。

## 开发环境

- 平台：Windows PowerShell
- 所有命令从仓库根目录运行
- Python 解释器：`.\.conda\python.exe`
- Python 版本：3.11.x
- 不得假设全局 `PATH` 中存在 Python
- 运行依赖位于 `requirements.txt`
- 测试依赖位于 `requirements-dev.txt`
- Git 跟踪文件中不得写死模型缓存路径；使用 `HF_HOME`

常用命令：

```powershell
& '.\.conda\python.exe' -m pip install -r requirements-dev.txt
& '.\.conda\python.exe' '.\scripts\run_hotpotqa_sample.py' --sample-size 1
& '.\.conda\python.exe' '.\scripts\build_hotpotqa_chunks.py' --sample-size 3
& '.\.conda\python.exe' '.\scripts\run_bm25.py' evaluate --chunks '.\data\processed\hotpotqa_bm25_sample100_chunks.jsonl' --corpus-scope global --output '.\results\bm25_hotpotqa_global_sample100.json'
& '.\.conda\python.exe' '.\scripts\run_bm25.py' query --sample-id '5a8b57f25542995d1e6f1371' --corpus-scope global --top-k 5 --show-gold
& '.\.conda\python.exe' '.\scripts\run_dense.py' build
& '.\.conda\python.exe' '.\scripts\run_dense.py' evaluate
& '.\.conda\python.exe' '.\scripts\run_hotpotqa_retrieval_experiment.py'
& '.\.conda\python.exe' '.\scripts\run_hotpotqa_generation_evaluation.py'
& '.\.conda\python.exe' '.\scripts\start_demo.py'
& '.\.conda\python.exe' -m pytest -q --basetemp '.\.test_tmp' -p no:cacheprovider
& '.\.conda\python.exe' -m uvicorn app.api:app --reload
```

## 技术方向

- 主数据集：HotpotQA；跨领域数据集：PubMedQA
- 配置文件放在 `configs/`
- 可复用实现放在 `src/`
- 可执行流水线放在 `scripts/`；HTTP 代码放在 `app/`
- 数据和索引等产物放在 `data/`；测量报告放在 `results/`
- 检索演进顺序：BM25 -> Dense FAISS -> Hybrid RRF -> Reranker
- 生成必须只使用检索证据并输出可追踪引用
- 检索、答案、引用、忠实度、延迟和成本必须分别评估

## 工程规则

- 保持 Python 3.11 兼容，并为公共接口提供类型标注
- 优先使用小型、经过测试的模块，而不是只在 Notebook 中实现
- 路径、随机种子、模型名和阈值必须可以配置
- 数据预处理和评估应尽可能保持确定性
- 使用 UTF-8 JSONL 保存便于检查的中间记录
- 保留样本、文档、标题、句子和证据来源信息
- 所有检索器使用稳定一致的指标定义，并按
  `(sample_id, title, sentence_id)` 对标准证据去重
- 除明确的评估或调试输出外，检索器不得看到答案文本或标准标签
- 基线切块不得拆分 HotpotQA 原始句子
- 行为发生变化时必须增加测试，并运行完整测试和烟雾命令
- 不得静默吞掉数据、模型或评估错误

## 安全与变更纪律

- 不得提交数据集、权重、索引、缓存、密钥或虚拟环境
- 不得删除 `data/raw/`，其中包含 HotpotQA 缓存
- 不得修改项目外的规划 PDF 或其他文件
- 覆盖实验结果前必须记录对应配置
- 不得随意重命名统一样本或文本块字段
- 保留与当前任务无关的用户改动
- 除非用户明确要求，否则不得初始化 Git、提交、推送或发布

## 阶段交接

每完成一个阶段：

1. 根据 `TODO.md` 检查验收标准
2. 在 `PROJECT_STATUS.md` 记录实际执行的命令和结果
3. 把下一阶段放在 `TODO.md` 的第一个章节
4. 只有长期规则或命令发生变化时才更新本文件

新窗口提示词：

```text
读取 PROJECT_STATUS.md、TODO.md、AGENTS.md 和可选的 AGENTS.local.md，继续 RAG 项目。
```
