import httpx
import pytest
from openai import BadRequestError

from rag.embed import DenseEmbedder


def bad_request():
    req = httpx.Request("POST", "http://x/v1/embeddings")
    return BadRequestError("EOF", response=httpx.Response(400, request=req), body=None)


@pytest.fixture
def embedder(monkeypatch):
    e = DenseEmbedder(batch=8)
    calls = []

    def fake_call(texts):  # fails whenever any input is longer than 1000 chars
        calls.append(len(texts))
        if any(len(t) > 1000 for t in texts):
            raise bad_request()
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(e, "_call", fake_call)
    return e, calls


def test_oversized_text_is_isolated_and_shrunk(embedder):
    e, calls = embedder
    vecs = e.embed(["a" * 10, "b" * 5000, "c" * 20])
    assert [v[0] for v in vecs] == [10, 625, 20]  # 5000 -> 2500 -> 1250 -> 625
    assert e.shrunk == 1
    assert calls[0] == 3  # first the whole batch


def test_normal_batch_single_call(embedder):
    e, calls = embedder
    e.embed(["x"] * 5)
    assert calls == [5] and e.shrunk == 0
