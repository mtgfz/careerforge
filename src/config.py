"""Central configuration, read from environment variables (.env supported)."""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
FAITHFULNESS_MIN = float(os.getenv("FAITHFULNESS_MIN", "0.95"))
OVERALL_MIN = float(os.getenv("OVERALL_MIN", "0.75"))

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

# Retrieval
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
K_PER_REQUIREMENT = 2
MAX_REQUIREMENTS = 12
MIN_RELEVANCE = 0.15
