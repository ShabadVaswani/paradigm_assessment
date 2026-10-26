"""Paths and settings. Everything can be overridden with environment variables, or with a `.env` file in the repo
root (gitignored; see `.env.example`). Variables already set in the environment win over the file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Put KEY=VALUE lines from `path` into os.environ (skipping keys already set), so both these settings and
    the Anthropic SDK see them. Blank lines, # comments and empty values are ignored (an empty ANTHROPIC_API_KEY
    would otherwise count as set); values may be quoted, and unquoted values may end with a " # comment"."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        if value:
            os.environ.setdefault(key, value)


load_dotenv()

HANDBOOK_URL = "https://paradigmiq.box.com/shared/static/grn1axnqf50npquo07x8r6yojb29g47l.pdf"
HANDBOOK_SHA256 = "0f81a75f987ef19e5fd18d85e1a07aa6bd7691a72a5723008f208c7cab7e2a57"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Paths:
    data: Path = ROOT / "data"
    pdf: Path = ROOT / "data" / "handbook.pdf"
    build: Path = ROOT / "data" / "build"
    images: Path = ROOT / "data" / "build" / "images"
    articles: Path = ROOT / "data" / "build" / "articles.json"
    image_index: Path = ROOT / "data" / "build" / "images.jsonl"
    chunks: Path = ROOT / "data" / "build" / "chunks.jsonl"
    # Image descriptions cost money to produce, so they are committed and reused.
    image_descriptions: Path = ROOT / "artifacts" / "image_descriptions.jsonl"
    # Embedding is slow on a laptop CPU, so vectors are committed too; only changed text is embedded again.
    vectors: Path = ROOT / "artifacts" / "embeddings" / "nomic-embed-text-v1.5.npz"


@dataclass(frozen=True)
class EmbedSettings:
    model_cache: Path = ROOT / "data" / "models"
    batch_size: int = int(_env("EMBED_BATCH_SIZE", "8"))
    threads: int = int(_env("EMBED_THREADS", "4"))
    pause_seconds: float = float(_env("EMBED_PAUSE_SECONDS", "0"))


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = int(_env("RETRIEVAL_TOP_K", "8"))
    candidates: int = int(_env("RETRIEVAL_CANDIDATES", "50"))  # from each of dense and BM25 before fusing
    rrf_k: int = int(_env("RETRIEVAL_RRF_K", "60"))
    # Cross-encoder reranking of the fused list: score the top `rerank_candidates`, keep the best `rerank_top_n`.
    # Model choice: notebooks/07_reranking.ipynb (decision 13). BAAI/bge-reranker-v2-m3-int8 ranked best there but
    # takes ~43 s per question on this laptop's CPU.
    rerank: bool = _env("RERANK", "on").lower() in ("on", "1", "true", "yes")
    rerank_model: str = _env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
    rerank_candidates: int = int(_env("RERANK_CANDIDATES", "30"))
    rerank_top_n: int = int(_env("RERANK_TOP_N", "5"))


@dataclass(frozen=True)
class ChunkSettings:
    min_tokens: int = int(_env("CHUNK_MIN_TOKENS", "400"))
    max_tokens: int = int(_env("CHUNK_MAX_TOKENS", "800"))
    overlap_sentences: int = int(_env("CHUNK_OVERLAP_SENTENCES", "2"))
    overlap_max_tokens: int = int(_env("CHUNK_OVERLAP_MAX_TOKENS", "120"))


@dataclass(frozen=True)
class LLMSettings:
    # "anthropic" = Claude API, "foundry" = Claude on Microsoft Foundry.
    provider: str = _env("LLM_PROVIDER", "anthropic")
    # When set, the API key is read from this Key Vault instead of the environment.
    key_vault_url: str = _env("AZURE_KEY_VAULT_URL", "")
    key_secret_name: str = _env("ANTHROPIC_KEY_SECRET_NAME", "anthropic-api-key")
    foundry_resource: str = _env("FOUNDRY_RESOURCE", "")
    image_model: str = _env("IMAGE_MODEL", "claude-sonnet-5-5")
    image_effort: str = _env("IMAGE_EFFORT", "medium")
    image_concurrency: int = int(_env("IMAGE_CONCURRENCY", "8"))


@dataclass(frozen=True)
class AnswerSettings:
    model: str = _env("ANSWER_MODEL", "claude-sonnet-5-5")
    effort: str = _env("ANSWER_EFFORT", "medium")
    top_k: int = int(_env("ANSWER_TOP_K", "10"))          # reranked excerpts passed to the model
    max_tool_rounds: int = int(_env("ANSWER_MAX_TOOL_ROUNDS", "4"))


@dataclass(frozen=True)
class JudgeSettings:
    model: str = _env("JUDGE_MODEL", "claude-opus-5-5")
    effort: str = _env("JUDGE_EFFORT", "high")


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    chunks: ChunkSettings = field(default_factory=ChunkSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    embed: EmbedSettings = field(default_factory=EmbedSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    answer: AnswerSettings = field(default_factory=AnswerSettings)
    judge: JudgeSettings = field(default_factory=JudgeSettings)


def load_settings() -> Settings:
    return Settings()
