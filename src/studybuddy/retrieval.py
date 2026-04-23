"""
Retrieval module.

ChromaDB'deki embedding'ler üzerinde semantic search yapar.
Ingestion offline, retrieval online — bu modül online yolun kalbidir.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from studybuddy import config
from studybuddy.ingestion import get_embedding_model

logger = logging.getLogger(__name__)

# E5 prefix'leri — asymmetric retrieval için (ingestion'da passage:, query'de query:)
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


# ─── Result Container ──────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    """Retrieval sonucu için structured container."""
    "'""chunk.source , chunk.source, -> pydantic olsaydı: chunk['source'] runtime'da patlar."""
    text: str               # temizlenmiş (passage: prefix'i kaldırılmış)
    source: str             # kaynak PDF dosyası
    page: int               # 0-indexed sayfa numarası
    chunk_id: str           # unique chunk identifier
    distance: float         # cosine/L2 distance (küçük = daha benzer)

    @property
    def page_display(self) -> int:
        """Kullanıcıya gösterim için 1-indexed sayfa."""
        return self.page + 1


# ─── Vector Store Singleton ────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_vectorstore(collection_name: str = "studybuddy") -> Chroma:
    """
    ChromaDB vector store'u tek sefer yükler, cache'ler.

    @lru_cache ile idempotent — 10 kez çağırılsa bir kez init olur.
    Bu pattern önemli: embedding model yüklemesi ~500MB belleğe çıkar,
    her retrieval'da tekrar yüklemek kabul edilemez.
    """
    logger.info(f"Vector store yükleniyor: collection={collection_name}")
    embeddings: HuggingFaceEmbeddings = get_embedding_model()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


# ─── Retrieval Function ────────────────────────────────────────────────
def retrieve(
    query: str,
    k: int | None = None,
    collection_name: str = "studybuddy",
) -> list[RetrievedChunk]:
    """
    Bir sorgu için ChromaDB'den top-k ilgili chunk'ı döndür.

    Args:
        query: Kullanıcı sorusu, ham metin (prefix'siz).
        k: Kaç chunk döndürülsün. None ise config.TOP_K kullanılır.
        collection_name: ChromaDB collection adı.

    Returns:
        RetrievedChunk listesi, distance'a göre sıralı (en benzer ilk).
    """
    k = k or config.TOP_K
    vectorstore = get_vectorstore(collection_name)

    # E5 asymmetric: query için "query: " prefix'i ekle
    prefixed_query = f"{QUERY_PREFIX}{query}"
    logger.debug(f"Retrieval query: {prefixed_query[:80]}...")

    raw_results = vectorstore.similarity_search_with_score(
        prefixed_query, k=k
    )

    # Pack into structured results + clean passage: prefix
    results: list[RetrievedChunk] = []
    for doc, distance in raw_results:
        text = doc.page_content
        if text.startswith(PASSAGE_PREFIX):
            text = text[len(PASSAGE_PREFIX):]

        results.append(
            RetrievedChunk(
                text=text,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", -1),
                chunk_id=doc.metadata.get("chunk_id", ""),
                distance=float(distance),
            )
        )

    logger.info(f"Retrieved {len(results)} chunks for query.")
    return results


# ─── CLI Entry Point ───────────────────────────────────────────────────
def main() -> None:
    """CLI: python -m studybuddy.retrieval "<soru>" """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print('Kullanım: python -m studybuddy.retrieval "<soru>"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    config.validate_config()
    chunks = retrieve(query)

    print(f"\n🔍 Query: {query}\n")
    print("=" * 80)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Match {i} (distance: {chunk.distance:.4f}) ---")
        print(f"📄 {chunk.source} | sayfa {chunk.page_display} | id {chunk.chunk_id}")
        print(f"📝 {chunk.text[:250]}...")


if __name__ == "__main__":
    main()