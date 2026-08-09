from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import app.api as api_module


def test_rag_service_initializes_once_under_concurrent_requests(monkeypatch) -> None:
    created = 0
    created_lock = threading.Lock()

    class FakeRAGService:
        def __init__(self, project_root) -> None:
            nonlocal created
            time.sleep(0.05)
            with created_lock:
                created += 1

    monkeypatch.setattr(api_module, "RAGService", FakeRAGService)
    monkeypatch.setattr(api_module, "_rag_service", None)

    with ThreadPoolExecutor(max_workers=8) as executor:
        services = list(executor.map(lambda _: api_module.get_rag_service(), range(8)))

    assert created == 1
    assert all(service is services[0] for service in services)
