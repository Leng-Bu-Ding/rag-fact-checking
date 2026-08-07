# AGENTS.md

Read `PROJECT_STATUS.md`, `TODO.md`, and this file before changing the project.

## Mission

Build a production-minded, interview-explainable RAG system. Favor measurable
retrieval quality, grounded answers, traceable evidence, and reproducible
experiments over feature count.

## Environment

- Platform: Windows PowerShell.
- Root: `D:\Projects\rag-fact-checking`.
- Conda manager: `D:\Tools\Anaconda3\Scripts\conda.exe`.
- Interpreter: `D:\Projects\rag-fact-checking\.conda\python.exe`.
- Python: 3.11.x.
- Never assume `python`, `python3`, or `py` is on `PATH`.
- Never use `.venv/`; it is incomplete.
- Runtime dependencies are in `requirements.txt`.
- Test dependencies are in `requirements-dev.txt`.
- Hugging Face model cache: `D:\Tools\HuggingFaceCache\hub`.

Common commands:

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

Run commands from the project root.

## Technical Direction

- Primary dataset: HotpotQA; cross-domain dataset: PubMedQA.
- Configuration belongs under `configs/`.
- Reusable implementation belongs under `src/`.
- Runnable pipelines belong under `scripts/`; HTTP code belongs under `app/`.
- Artifacts belong under `data/` and measured reports under `results/`.
- Retrieval progression: BM25 -> dense FAISS -> hybrid RRF -> reranking.
- Generation must use retrieved evidence and emit traceable citations.
- Evaluate retrieval, answers, citations, faithfulness, latency, and cost
  separately.

## Engineering Rules

- Keep Python 3.11 compatibility and type-hint public interfaces.
- Prefer small tested modules over notebook-only implementations.
- Keep paths, seeds, model names, and thresholds configurable.
- Make preprocessing and evaluation deterministic where practical.
- Use UTF-8 JSONL for inspectable intermediate records.
- Preserve sample, document, title, sentence, and evidence provenance.
- Keep retrieval metric definitions stable across retrievers and deduplicate
  gold evidence by `(sample_id, title, sentence_id)`.
- Never expose answer text or gold labels to a retriever except in explicit
  evaluation or debugging output.
- Do not split original HotpotQA sentences during baseline chunking.
- Add tests for behavioral changes and run tests plus a smoke command.
- Do not silently catch data, model, or evaluation errors.

## Safety And Change Discipline

- Do not commit datasets, weights, indexes, caches, secrets, or environments.
- Do not delete `data/raw/`; it contains the HotpotQA cache.
- Do not modify planning PDFs or files outside this project.
- Do not overwrite experiment results without recording their configuration.
- Do not casually rename unified sample or chunk fields.
- Preserve unrelated user changes.
- Do not initialize Git, commit, push, or publish unless the user asks.

## Stage Handoff

At every completed stage:

1. Verify the acceptance criteria in `TODO.md`.
2. Record measured commands and outcomes in `PROJECT_STATUS.md`.
3. Make the next stage the first section of `TODO.md`.
4. Update this file only when durable rules or commands change.

New-window prompt:

```text
璇诲彇 PROJECT_STATUS.md銆乀ODO.md 鍜?AGENTS.md锛岀户缁?RAG 椤圭洰銆?```
