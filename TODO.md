# TODO

Last updated: 2026-08-03

## Next Milestone: HotpotQA Scale And Answer Quality

Goal: preserve the completed, reproducible 100-sample retrieval benchmark while
scaling data and replacing the weak local generator.

- [ ] Run a larger HotpotQA validation subset and record CPU time, memory, and
  index size before attempting the full split.
- [ ] Replace or augment FLAN-T5-small with a stronger instruction model/API and
  compare Exact Match, Token F1, citations, latency, and cost.
- [ ] Calibrate semantic abstention on supported and unsupported question sets.
- [ ] Add claim-level faithfulness evaluation instead of treating citation
  presence as proof of faithfulness.
- [ ] Add repeatable demo stop/status commands and structured request logs.
- [ ] Make an explicit local-only versus public-hosting decision.
- [ ] After HotpotQA scaling, transfer the same interfaces and reports to
  PubMedQA.

## Completed: HotpotQA 100-Sample Interview MVP

- [x] Deterministic 20-development / 80-test split with a recorded hash.
- [x] Common BM25, Dense, Hybrid, and Reranker interfaces.
- [x] Leakage-free RRF grid tuning on development IDs only.
- [x] Cross-Encoder reranking with measured CPU latency.
- [x] Hit, Recall, MRR, Complete, gold-document recall, and novelty-aware
  fact-NDCG metrics.
- [x] Per-question improvement/regression records for error analysis.
- [x] Grounded-answer Exact Match and Token F1 evaluation.
- [x] Citation presence, validity, precision, and gold-fact recall evaluation.
- [x] Explicit rejection for clear out-of-domain questions.
- [x] Cross-Encoder integrated into the FastAPI/browser demo.
- [x] Reproducible JSON experiment reports with source and split hashes.
- [x] 39 passing automated tests plus compilation and real-model smoke checks.

## Stage Acceptance

The 100-sample HotpotQA milestone is accepted because retrieval, generation,
citations, abstention, latency, and error cases are measured separately on a
fixed split, and the same pipeline is available through the local demo.

This does not claim that the full HotpotQA dataset or public production
deployment is complete. Those are the next scale and operations milestones.
