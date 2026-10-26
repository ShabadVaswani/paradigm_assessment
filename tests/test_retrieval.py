import numpy as np

from handbook_qa.config import EmbedSettings
from handbook_qa.embeddings import VectorStore, embed_items, text_hash
from handbook_qa.retrieval import (
    BM25,
    Item,
    Retriever,
    reciprocal_rank_fusion,
    tokenize,
)


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


class KeywordReranker:
    """Scores a text by how many times it contains the word 'answer'."""

    def score(self, query, texts):
        return np.array([t.count("answer") for t in texts], dtype=np.float32)


def reranking_retriever():
    texts = [f"doc {i} filler" for i in range(6)] + ["the answer answer answer", "an answer"]
    vectors = np.eye(8, dtype=np.float32)[:, :8]
    query = np.linspace(1.0, 0.1, 8)  # dense order: doc 0 first, the "answer" docs last
    return Retriever([Item(f"c{i}", "chunk", t, {}) for i, t in enumerate(texts)], vectors, lambda q: query,
                     candidates=10, rrf_k=60, reranker=KeywordReranker(), rerank=True,
                     rerank_candidates=8, rerank_top_n=2)


def test_reranker_reorders_fused_candidates_and_keeps_top_n():
    hits = reranking_retriever().search("question")
    assert [h.item.id for h in hits] == ["c6", "c7"]
    assert hits[0].rerank_score == 3 and hits[0].fused_rank > 2  # promoted from low in the fused list


def test_reranking_can_be_switched_off_per_call():
    hits = reranking_retriever().search("question", k=3, rerank=False)
    assert [h.item.id for h in hits] == ["c0", "c1", "c2"]
    assert all(h.rerank_score is None for h in hits)


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


def test_dynamic_selection_keeps_passing_candidates_within_the_clamp(tmp_path):
    from handbook_qa.retrieval import DynamicSelection

    sel = DynamicSelection(cutoff=1.0, candidates=8, min_keep=3, max_keep=5, model="fake")
    assert sel.count([3, 2, 1, 0, -1]) == 3        # three pass
    assert sel.count([0.5, 0.2]) == 3              # none pass: floor of 3
    assert sel.count([9] * 8) == 5                 # all pass: cap of 5

    path = tmp_path / "frozen.json"
    path.write_text('{"cutoff": -3.5, "candidates_reranked": 50, "min_kept": 3, "max_kept": 15, '
                    '"model": "Xenova/ms-marco-MiniLM-L-6-v2", "rule": "..."}')
    assert DynamicSelection.load(path) == DynamicSelection(-3.5, 50, 3, 15, "Xenova/ms-marco-MiniLM-L-6-v2")


def test_search_dynamic_uses_frozen_settings_and_checks_the_model():
    from handbook_qa.retrieval import DynamicSelection

    retriever = reranking_retriever()
    retriever.reranker.model_name = "fake"
    hits = retriever.search_dynamic("question", DynamicSelection(1.0, 8, 1, 5, "fake"))
    assert [h.item.id for h in hits] == ["c6", "c7"]          # only the two "answer" docs score >= 1
    assert retriever.rerank_candidates == 8                    # restored afterwards

    try:
        retriever.search_dynamic("question", DynamicSelection(1.0, 8, 1, 5, "another-model"))
        raise AssertionError("expected a model mismatch error")
    except RuntimeError as error:
        assert "tuned for another-model" in str(error)
