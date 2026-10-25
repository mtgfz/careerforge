"""Central configuration, read from environment variables (.env supported)."""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Judge (Critic) model; falls back to the main LLM when left empty.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "").lower()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
# Per-request timeout (seconds) and SDK retries for every chat model. Several models can be
# listed in LLM_MODEL (comma-separated, in order): the next one is tried when a model fails.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))

# "Light" model for job analysis, cover letter and interview questions (empty = main LLM).
# Free quotas are per model: moving these calls off the writer model saves its daily quota.
LIGHT_PROVIDER = os.getenv("LIGHT_PROVIDER", "").lower()
LIGHT_MODEL = os.getenv("LIGHT_MODEL", "")

# Backup judge used when the judge keeps failing; falls back to the main LLM when left empty.
JUDGE_FALLBACK_PROVIDER = os.getenv("JUDGE_FALLBACK_PROVIDER", "").lower()
JUDGE_FALLBACK_MODEL = os.getenv("JUDGE_FALLBACK_MODEL", "")

# Structured-output calls (tool calling) fail now and then on free models: retry before giving up.
STRUCTURED_RETRIES = int(os.getenv("STRUCTURED_RETRIES", "3"))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
FAITHFULNESS_MIN = float(os.getenv("FAITHFULNESS_MIN", "0.95"))
OVERALL_MIN = float(os.getenv("OVERALL_MIN", "0.75"))

# --- Speed knobs -----------------------------------------------------------------------------
# Stop the Writer<->Critic loop when an extra iteration improves the overall score by less than this.
PLATEAU_DELTA = float(os.getenv("PLATEAU_DELTA", "0.02"))
# True (default, used by the ablation): re-judge the CV after pruning. False (web app): skip that
# extra LLM call; after pruning nothing unsupported is left, so faithfulness is 1.0 by construction.
REJUDGE_AFTER_PRUNE = os.getenv("REJUDGE_AFTER_PRUNE", "true").lower() in ("1", "true", "yes")
# Generate the cover letter and the interview questions concurrently.
PARALLEL_EXTRAS = os.getenv("PARALLEL_EXTRAS", "true").lower() in ("1", "true", "yes")
# Knowledge bases (embedded documents) kept in memory, and how long GitHub profiles are cached.
KB_CACHE_SIZE = int(os.getenv("KB_CACHE_SIZE", "8"))
GITHUB_CACHE_SECONDS = int(os.getenv("GITHUB_CACHE_SECONDS", "600"))

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

# Retrieval
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
K_PER_REQUIREMENT = 2
MAX_REQUIREMENTS = 12
MIN_RELEVANCE = 0.15
