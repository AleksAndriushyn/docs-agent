from core.pricing import Usage


def test_cost_and_diff():
    u = Usage()
    u.add("claude-opus-5-5", 1_000_000, 100_000, cache_read=1_000_000, cache_write=0)
    assert abs(u.cost() - (4.0 + 2.0 + 0.2)) < 1e-9
    before = u.copy()
    u.add("claude-opus-5-5", 500_000, 0)
    d = u - before
    assert d.input == 500_000 and d.calls == 1 and abs(d.cost() - 2.0) < 1e-9


def test_local_model_is_free():
    u = Usage()
    u.add("qwen3.6", 1000, 100)
    assert u.cost() is None and u.input == 1000
