import numpy as np

from handbook_qa.config import EmbedSettings
from handbook_qa.embeddings import VectorStore, embed_items, text_hash
from handbook_qa.retrieval import BM25, Item, Retriever, reciprocal_rank_fusion, tokenize


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("How do refunds and policies work for the #merch channel?") == \
        ["refund", "policy", "work", "merch", "channel"]


def test_bm25_prefers_documents_with_rare_query_terms():
    docs = [tokenize(t) for t in ["merch requests go to the brand team",
                                  "the team plans offsites and the team travels",
                                  "expenses and travel policy"]]
    scores = BM25(docs).scores(tokenize("merch request"))
    assert scores.argmax() == 0 and scores[1] == 0


def test_rrf_rewards_items_found_by_both_lists():
    fused = reciprocal_rank_fusion([[0, 1, 2], [2, 3, 0]], k=60)
    # 0 and 2 are in both lists (ranks 1+3 and 3+1); 1 and 3 are 2nd in one list only.
    assert fused[0] == fused[2] == 1 / 61 + 1 / 63
    assert fused[1] == fused[3] == 1 / 62
    assert fused[0] > fused[1]


def make_retriever(texts, vectors, query_vector):
    items = [Item(f"c{i}", "chunk", t, {}) for i, t in enumerate(texts)]
    return Retriever(items, np.array(vectors, dtype=np.float32), lambda q: np.array(query_vector, dtype=np.float32),
                     candidates=10, rrf_k=60)


def test_hybrid_search_combines_dense_and_keyword_matches():
    texts = ["Slack channel #team-marketing for marketing questions",   # keyword match only
             "Ask the brand designers for artwork and swag",            # semantic match only
             "Quarterly planning and OKRs"]                             # neither
    vectors = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    retriever = make_retriever(texts, vectors, query_vector=[0.9, 0.1, 0.0])
    hits = retriever.search("#team-marketing", k=3)
    ids = [h.item.id for h in hits]
    assert set(ids[:2]) == {"c0", "c1"}
    assert hits[0].dense_rank is not None or hits[0].bm25_rank is not None
    c0 = next(h for h in hits if h.item.id == "c0")
    assert c0.bm25_rank == 1


class FakeEmbedder:
    calls = 0

    def __init__(self, settings):
        pass

    def embed_documents(self, texts):
        FakeEmbedder.calls += 1
        return np.array([[len(t), 1.0, 0.0] for t in texts], dtype=np.float32)


def test_embed_items_only_embeds_new_or_changed_text(tmp_path, monkeypatch):
    monkeypatch.setattr("handbook_qa.embeddings.DIMENSIONS", 3)
    path = tmp_path / "vectors.npz"
    settings = EmbedSettings(model_cache=tmp_path, batch_size=2, threads=1)
    first = embed_items([("a", "alpha"), ("b", "beta"), ("c", "gamma")], path, settings, FakeEmbedder)
    assert first == {"embedded": 3, "reused": 0}

    second = embed_items([("a", "alpha"), ("b", "beta changed"), ("d", "delta")], path, settings, FakeEmbedder)
    assert second == {"embedded": 2, "reused": 1}

    store = VectorStore.load(path)
    assert store.ids == ["a", "b", "d"]  # "c" dropped, order follows the items
    assert store.hashes[1] == text_hash("beta changed")
    assert store.vectors.shape == (3, 3)
