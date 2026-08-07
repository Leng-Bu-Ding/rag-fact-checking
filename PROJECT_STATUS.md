# Project Status

Last updated: 2026-08-03

## Current Stage

The measured HotpotQA 100-sample interview MVP is complete. It now includes a
fixed development/test split, leakage-free Hybrid RRF tuning, Cross-Encoder
reranking, multi-hop retrieval metrics, grounded-answer evaluation, explicit
out-of-domain abstention, FastAPI, and a browser UI.

The next milestone is scale and model quality: run a larger HotpotQA corpus,
replace the weak FLAN-T5-small generator, then transfer the same framework to
PubMedQA.

## Verified Local Demo

- Web UI: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs
- Knowledge chunks: 1,778 from 100 HotpotQA validation samples.
- Embedding model: sentence-transformers/all-MiniLM-L6-v2, 384 dimensions.
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2, local CPU execution.
- Generator: google/flan-t5-small, local CPU execution.
- Model cache: D:\Tools\HuggingFaceCache\hub.
- Final automated test run: 39 passed in 3.15s.
- Python compilation check: passed.
- Verified supported question returns the correct answer with two evidence
  citations in about 403 ms after model warm-up.
- Verified unsupported question returns an explicit insufficient-evidence answer.

## Fixed-Split Retrieval Experiment

Input: 100 questions, 1,778 chunks, and 243 unique supporting facts. The split
is deterministic: 20 development questions for RRF parameter selection and 80
untouched test questions. Split SHA-256:
08c3e1aa1a8fe2844f899684db7e1cfd1d2f0915e31edafe1b17c8a795891660.

| System | Hit@1 | Hit@5 | Recall@5 | Complete@5 | Recall@10 | Complete@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.8000 | 0.9625 | 0.7165 | 0.4375 | 0.8258 | 0.6375 | 0.8748 |
| Dense FAISS | 0.7750 | 0.9500 | 0.6965 | 0.4250 | 0.8104 | 0.5875 | 0.8614 |
| Hybrid RRF | 0.8375 | 1.0000 | 0.7404 | 0.4500 | 0.8413 | 0.6500 | 0.9004 |
| Cross-Encoder reranked | 0.9000 | 0.9875 | 0.7946 | 0.5750 | 0.9094 | 0.8000 | 0.9378 |

RRF selected candidate_k=10 and rrf_k=10 on development data only.
Cross-Encoder reranking adds about 428 ms per query on CPU. At Recall@10 it
improved 14 test questions, regressed 2, and tied 64 relative to Hybrid.

The complete report, including per-question evidence and error cases, is
results/hotpotqa_retrieval_core_sample100.json.

## Grounded-Answer Experiment

The same 80 test questions were generated from the top three reranked evidence
chunks.

| Metric | Result |
|---|---:|
| Exact Match | 0.3375 |
| Token F1 | 0.4256 |
| Has citation | 1.0000 |
| Citation validity | 1.0000 |
| Citation precision | 0.7333 |
| Citation gold-fact recall | 0.5925 |
| Generation latency | 300 ms/query |

This deliberately separates retrieval quality from answer quality. Retrieval
is strong for the current benchmark, while FLAN-T5-small is the main quality
bottleneck. The complete report is
results/hotpotqa_generation_sample100_test80.json.

## Reproducibility And Safety Decisions

- Use D:\Projects\rag-fact-checking\.conda\python.exe; never assume Python is
  globally available.
- Retrieval never indexes answer text or gold labels.
- Gold facts are deduplicated by (sample_id, title, sentence_id).
- BM25, Dense, Hybrid, and Reranker are evaluated on identical test IDs.
- RRF parameters are selected on development IDs only.
- Reports record data hashes, split IDs, configuration, per-query metrics,
  latency, improvements, and regressions.
- Equal scores are resolved by deterministic chunk ID ordering.
- The service rejects clear out-of-domain questions with no lexical connection
  to retrieved evidence.
- The service remains local-only and has not been exposed to the public
  internet.

## Remaining Risks

- The measured corpus is a 100-sample benchmark, not the full HotpotQA
  validation corpus.
- FLAN-T5-small answer quality is not production-grade.
- Lexical out-of-domain rejection handles clear misses but is not a calibrated
  semantic confidence model.
- Public deployment still requires hosting, authentication, monitoring, and
  cost-control decisions.
- The repository has not been initialized as Git because the user has not
  requested it.
