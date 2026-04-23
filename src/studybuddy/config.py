"""
Configuration module for StudyBuddy RAG.

Tek noktadan ayar yönetimi (single source of truth pattern).
Tüm path, model ve hyperparameter değerleri buradan okunur.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ─── .env yükleme ──────────────────────────────────────────────────────
# Proje kökündeki .env dosyasını okur, environment variable'lara yükler.
# load_dotenv() aynı dizinde .env arar, üst dizinlere de bakabilir.
load_dotenv()


# ─── Path'ler ──────────────────────────────────────────────────────────
# __file__: bu dosyanın yolu (/Users/.../src/studybuddy/config.py)
# parent: src/studybuddy/
# parent.parent: src/
# parent.parent.parent: proje kökü
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
# CHROMA_PERSIST_DIR env var relative olabilir, her zaman proje köküne göre absolute yap
_chroma_env = os.getenv("CHROMA_PERSIST_DIR")
if _chroma_env:
    CHROMA_DIR: Path = (PROJECT_ROOT / _chroma_env).resolve()
else:
    CHROMA_DIR: Path = (DATA_DIR / "chroma").resolve()

# Klasörler yoksa oluştur (idempotent, her zaman güvenli)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ─── API Keys ──────────────────────────────────────────────────────────
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")


# ─── Model İsimleri ────────────────────────────────────────────────────
# Multilingual-e5: Türkçe dahil 100+ dilde iyi çalışan embedding modeli.
# Küçük versiyon (small) seçtik: 384-dim, hızlı, yeterince kaliteli.
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)

# Llama 3.3 70B: Groq'ta ücretsiz, Türkçe iyi, hızlı inference.
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


# ─── RAG Hyperparametreleri ────────────────────────────────────────────
# Chunking
CHUNK_SIZE: int = 500         # karakter cinsinden chunk boyutu
CHUNK_OVERLAP: int = 100      # ardışık chunk'lar arasında kesişim

# Retrieval
TOP_K: int = 5                # similarity search'ten kaç chunk döndürülecek

# LLM sampling
TEMPERATURE: float = 0.1      # düşük = deterministik, yüksek = yaratıcı
MAX_TOKENS: int = 1024        # cevapta max üretilecek token sayısı


# ─── Sanity Check ──────────────────────────────────────────────────────
def validate_config() -> None:
    """Kritik ayarların dolu olduğunu kontrol et, yoksa anlamlı hata ver."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY bulunamadı. .env dosyasını kontrol et. "
            "Örnek: GROQ_API_KEY=gsk_xxxxx"
        )


if __name__ == "__main__":
    # `python -m studybuddy.config` ile çalıştırıldığında
    # config'in doğru yüklendiğini doğrulamak için.
    validate_config()
    print("✓ Config OK")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  UPLOADS_DIR:  {UPLOADS_DIR}")
    print(f"  CHROMA_DIR:   {CHROMA_DIR}")
    print(f"  EMBEDDING:    {EMBEDDING_MODEL}")
    print(f"  LLM:          {LLM_MODEL}")
    print(f"  GROQ_KEY:     {'***' + GROQ_API_KEY[-4:] if GROQ_API_KEY else 'MISSING'}")