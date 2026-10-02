from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHUNK_WORDS = 160
DEFAULT_OVERLAP_WORDS = 30
DEFAULT_TOP_K = 5
# Keep the original calibrated value for local excerpts; LLM mode assesses evidence.
DEFAULT_MIN_RELEVANCE = 0.16
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 300
MAX_TEXT_CHARACTERS = 2_000_000
MAX_CHUNKS = 3000
