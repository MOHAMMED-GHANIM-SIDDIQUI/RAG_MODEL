from pathlib import Path
import os


APP_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = APP_DIR / "data"
PDF_PATH = DATA_DIR / "CONSTITUTION.pdf"
INDEX_DIR = APP_DIR / "faiss-index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "mixtral-8x7b-32768"


def get_setting(name: str, default: str | None = None, secrets=None) -> str | None:
    if secrets is not None and name in secrets:
        return secrets[name]
    return os.getenv(name, default)


def is_enabled(name: str, default: bool = False, secrets=None) -> bool:
    value = str(get_setting(name, str(default), secrets=secrets)).lower()
    return value in {"1", "true", "yes", "on"}
