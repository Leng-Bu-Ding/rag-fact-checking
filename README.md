# Trustworthy RAG for Fact-Checking QA

A local, evidence-grounded RAG system for multi-hop fact checking. It combines
BM25 exact matching, MiniLM semantic retrieval, persistent FAISS search,
Hybrid RRF, Cross-Encoder reranking, grounded generation, abstention, and
traceable citations.

## Run The Local Demo

From PowerShell:

    & '.\.conda\python.exe' '.\scripts\start_demo.py'

Then open:

- Web UI: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The first model load can take tens of seconds. A verified warm request takes
about 403 ms with Cross-Encoder reranking on CPU.

## What Is Implemented

- deterministic HotpotQA loading, sentence-aware chunking, and UTF-8 JSONL
- sample, document, title, sentence, gold-evidence, and citation provenance
- BM25 and normalized MiniLM/FAISS retrieval
- Hybrid RRF with development-only parameter tuning
- Cross-Encoder reranking using ms-marco-MiniLM-L-6-v2
- Hit, Recall, MRR, Complete, gold-document recall, and fact-NDCG metrics
- fixed 20-development / 80-test split with a recorded SHA-256
- per-question improvement/regression error records
- grounded-answer Exact Match, Token F1, and citation evaluation
- explicit abstention for clear questions unsupported by the corpus
- FastAPI endpoints and an interactive browser evidence UI
- 39 passing automated tests

## Architecture

    Question
       |-- BM25 keyword ranking -----------|
       |-- MiniLM -> FAISS cosine ranking--|-> Hybrid RRF
                                                |
                                                v
                                         Cross-Encoder
                                                |
                                                v
                                  evidence -> answer -> citations
                                                |
                                      insufficient? -> abstain

Important implementation files:

    configs/hotpotqa_retrieval.yaml
    src/retrieval/base.py
    src/retrieval/bm25.py
    src/retrieval/dense.py
    src/retrieval/composite.py
    src/retrieval/reranker.py
    src/evaluation/retrieval.py
    src/evaluation/answers.py
    scripts/run_hotpotqa_retrieval_experiment.py
    scripts/run_hotpotqa_generation_evaluation.py
    src/services/rag.py
    app/api.py

## Reproduce The Experiments

Use only the project-local Python 3.11 environment.

    & '.\.conda\python.exe' '.\scripts\run_hotpotqa_retrieval_experiment.py'
    & '.\.conda\python.exe' '.\scripts\run_hotpotqa_generation_evaluation.py'
    & '.\.conda\python.exe' -m pytest -q --basetemp '.\.test_tmp' -p no:cacheprovider

Models are cached under D:\Tools\HuggingFaceCache\hub and are loaded offline
after their initial download.

## Fixed-Split HotpotQA Results

The input has 100 validation questions, 1,778 chunks, and 243 unique supporting
facts. RRF parameters are selected on 20 development questions; the following
numbers are from the untouched 80-question test set.

| System | Hit@1 | Recall@5 | Complete@5 | Recall@10 | Complete@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.8000 | 0.7165 | 0.4375 | 0.8258 | 0.6375 | 0.8748 |
| Dense FAISS | 0.7750 | 0.6965 | 0.4250 | 0.8104 | 0.5875 | 0.8614 |
| Hybrid RRF | 0.8375 | 0.7404 | 0.4500 | 0.8413 | 0.6500 | 0.9004 |
| Cross-Encoder | 0.9000 | 0.7946 | 0.5750 | 0.9094 | 0.8000 | 0.9378 |

Cross-Encoder reranking is the strongest retrieval system, but costs about
428 ms per query on CPU.

For grounded answers over those 80 questions:

| Metric | Result |
|---|---:|
| Exact Match | 0.3375 |
| Token F1 | 0.4256 |
| Citation validity | 1.0000 |
| Citation precision | 0.7333 |
| Citation gold-fact recall | 0.5925 |

The separation is intentional: retrieval is strong on this benchmark, while
FLAN-T5-small answer generation is still the main quality bottleneck.

## Reports

- results/hotpotqa_retrieval_core_sample100.json
- results/hotpotqa_generation_sample100_test80.json
- PROJECT_STATUS.md
- TODO.md

## Honest Scope

This completes the measured 100-sample HotpotQA interview MVP, not the full
HotpotQA validation corpus and not a public production deployment. The next
work is larger-scale HotpotQA evaluation, a stronger answer model, calibrated
semantic abstention, and deployment decisions. PubMedQA comes after this
HotpotQA scaling step.
