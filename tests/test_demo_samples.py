from src.services.rag import order_demo_samples


def test_order_demo_samples_puts_verified_examples_first_without_dropping_others() -> None:
    samples = [
        {"sample_id": "a", "question": "A?"},
        {"sample_id": "b", "question": "B?"},
        {"sample_id": "c", "question": "C?"},
    ]

    ordered = order_demo_samples(samples, ["c", "missing", "a"])

    assert [item["sample_id"] for item in ordered] == ["c", "a", "b"]
