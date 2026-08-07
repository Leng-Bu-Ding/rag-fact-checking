# 可信 RAG：面向多跳事实核验问答

这是一个在本地运行、以证据为中心的多跳问答 RAG 项目。系统组合了
BM25 关键词检索、MiniLM 语义检索、FAISS 向量索引、Hybrid RRF 融合、
Cross-Encoder 重排序、基于证据的答案生成、拒答机制和可追溯引用。

## 项目已经实现什么

- 确定性的 HotpotQA 数据加载、句子级切块和 UTF-8 JSONL 中间数据
- 样本、文档、标题、句子、标准证据和答案引用的完整来源信息
- BM25 检索和归一化 MiniLM/FAISS 向量检索
- 只使用开发集选择参数的 Hybrid RRF
- 使用 `ms-marco-MiniLM-L-6-v2` 的 Cross-Encoder 重排序
- Hit、Recall、MRR、Complete、标准文档召回和 fact-NDCG 指标
- 固定的 20 条开发集 / 80 条测试集划分及其 SHA-256
- 逐问题的提升、退化和错误分析记录
- 答案 Exact Match、Token F1 和引用质量评估
- 面对语料库不支持的问题时明确拒答
- FastAPI 接口和可交互的浏览器证据页面
- 检索、生成、评估和配置方面的自动化测试

## 系统流程

    用户问题
       |-- BM25 关键词排序 --------------|
       |-- MiniLM -> FAISS 向量排序 ------|-> Hybrid RRF 融合
                                                |
                                                v
                                         Cross-Encoder 重排序
                                                |
                                                v
                                      证据 -> 答案 -> 引用
                                                |
                                      证据不足？ -> 拒答

## 从零建立环境

需要提前安装 Git、Conda，并使用 PowerShell。第一次下载数据和模型时需要
联网。在仓库根目录执行：

    conda create --prefix .\.conda python=3.11 -y
    conda activate .\.conda
    python -m pip install -r requirements-dev.txt

Git 仓库不包含虚拟环境、数据集、模型权重、向量索引或生成的实验报告；
这些内容需要在本机生成。

Hugging Face 默认使用自己的标准缓存目录。如果希望把模型放在其他磁盘，
请在运行模型命令前设置 `HF_HOME`：

    $env:HF_HOME = 'D:\path\to\HuggingFaceCache'

这是机器自己的设置，不是项目固定路径。已有模型会直接复用，缺少的模型
会在第一次使用时下载。

如果模型已经完整缓存，并且希望彻底禁止联网检查，可以额外设置：

    $env:HF_HUB_OFFLINE = '1'

在所需模型尚未下载完整之前，不要开启离线模式。

## 构建并复现 100 样本实验

原始数据来自
[Hugging Face 上的 HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)。
按以下顺序执行：

    & '.\.conda\python.exe' '.\scripts\build_hotpotqa_chunks.py' --sample-size 100 --output '.\data\processed\hotpotqa_bm25_sample100_chunks.jsonl'
    & '.\.conda\python.exe' '.\scripts\run_bm25.py' evaluate --chunks '.\data\processed\hotpotqa_bm25_sample100_chunks.jsonl' --corpus-scope global --output '.\results\bm25_hotpotqa_global_sample100.json'
    & '.\.conda\python.exe' '.\scripts\run_dense.py' build
    & '.\.conda\python.exe' '.\scripts\run_dense.py' evaluate
    & '.\.conda\python.exe' '.\scripts\run_hotpotqa_retrieval_experiment.py'
    & '.\.conda\python.exe' '.\scripts\run_hotpotqa_generation_evaluation.py'

生成的数据、索引和详细 JSON 报告位于 `data/` 和 `results/`，并被
`.gitignore` 排除，不会上传到 GitHub。

## 启动本地演示

先完成数据切块和 Dense 索引构建，然后执行：

    & '.\.conda\python.exe' '.\scripts\start_demo.py'

浏览器访问：

- Web 页面：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

模型首次载入可能需要几十秒。CPU 预热后，一次包含 Cross-Encoder 重排序的
已验证请求约耗时 403 ms。

## 运行测试

    & '.\.conda\python.exe' -m pytest -q --basetemp '.\.test_tmp' -p no:cacheprovider

当前完整测试结果：`42 passed`。

## HotpotQA 固定划分结果

实验输入包括 100 条验证集问题、1,778 个文本块和 243 条去重后的标准证据。
RRF 参数只在 20 条开发集问题上选择，下表来自未参与调参的 80 条测试集。

| 系统 | Hit@1 | Recall@5 | Complete@5 | Recall@10 | Complete@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.8000 | 0.7165 | 0.4375 | 0.8258 | 0.6375 | 0.8748 |
| Dense FAISS | 0.7750 | 0.6965 | 0.4250 | 0.8104 | 0.5875 | 0.8614 |
| Hybrid RRF | 0.8375 | 0.7404 | 0.4500 | 0.8413 | 0.6500 | 0.9004 |
| Cross-Encoder | 0.9000 | 0.7946 | 0.5750 | 0.9094 | 0.8000 | 0.9378 |

Cross-Encoder 是当前最强的检索方案，但在 CPU 上每条问题增加约 428 ms。

80 条测试问题的答案生成结果：

| 指标 | 结果 |
|---|---:|
| Exact Match | 0.3375 |
| Token F1 | 0.4256 |
| 引用有效率 | 1.0000 |
| 引用精确率 | 0.7333 |
| 标准证据引用召回率 | 0.5925 |

这些结果刻意把检索质量和答案质量分开衡量：当前检索效果较强，
FLAN-T5-small 的答案生成能力仍是主要瓶颈。

## 重要目录和文件

- `configs/`：可复现且不依赖具体电脑的实验配置
- `src/`：数据、检索、生成和评估的可复用实现
- `scripts/`：可以直接执行的数据及实验流水线
- `app/`：FastAPI 服务和浏览器页面
- `tests/`：自动化行为测试和配置测试
- `PROJECT_STATUS.md`：已验证结果和工程决策
- `TODO.md`：下一阶段计划与验收标准

## 项目边界

当前完成的是经过测量的 HotpotQA 100 样本面试版 MVP，不是完整 HotpotQA
验证集实验，也不是已经部署到公网的生产系统。下一阶段是扩大 HotpotQA
评测规模、替换更强的答案模型、校准语义拒答机制，再把同一框架迁移到
PubMedQA。
