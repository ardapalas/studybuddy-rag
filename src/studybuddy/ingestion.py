"""
Document ingestion pipeline.

PDF dosyalarını okuyup chunk'lara bölerek ChromaDB'ye embedding olarak yükler.
Bu modül offline/batch çalışır — her PDF bir kere ingest edilir.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader,PDFPlumberLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from studybuddy import config

logger = logging.getLogger(__name__)


# ─── Embedding Modeli ──────────────────────────────────────────────────
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Multilingual-e5-small embedding modelini yükler.

    E5 modelleri asymmetric: passages ve queries farklı prefix ister.
    Bu fonksiyon 'passage:' prefix'ini otomatik ekleyecek şekilde konfigüre eder.
    Retrieval sırasında 'query:' prefix'i ayrıca retrieval.py'de eklenecek.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # MPS de çalışır ama CPU daha stable
        encode_kwargs={
            "normalize_embeddings": True,  # cosine similarity için normalize
        },
    )


# ─── PDF Parsing + Chunking ────────────────────────────────────────────
def load_and_chunk_pdf(pdf_path: Path, loader_type) -> list[Document]:
    """
    PDF'i yükler, RecursiveCharacterTextSplitter ile chunk'lara böler.

    Her chunk LangChain Document objesi olur:
      - page_content: chunk metni (passage: prefix'li)
      - metadata: {source, page, chunk_id}
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF bulunamadı: {pdf_path}")

    logger.info(f"PDF yükleniyor: {pdf_path.name}")
    if loader_type == "pdfplumber":
        loader = PDFPlumberLoader(str(pdf_path))
    else:
        loader = PyPDFLoader(str(pdf_path))
    pages: list[Document] = loader.load()
    logger.info(f"  → {len(pages)} sayfa okundu")

    # Splitter: semantic boundary-aware, paragraf/cümle sınırlarını tercih eder
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # büyükten küçüğe dener
        length_function=len,
    )

    chunks: list[Document] = splitter.split_documents(pages)
    logger.info(f"  → {len(chunks)} chunk üretildi")

    # E5 prefix: her chunk'ın başına 'passage: ' ekle
    # Ayrıca unique chunk_id ekleyelim (metadata filtering için)
    file_id = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
    for i, chunk in enumerate(chunks):
        chunk.page_content = f"passage: {chunk.page_content}"
        chunk.metadata["chunk_id"] = f"{file_id}_{i:04d}"
        chunk.metadata["source"] = pdf_path.name

    return chunks


# ─── ChromaDB'ye Yükleme ───────────────────────────────────────────────
def ingest_pdf(
    pdf_path: Path,
    collection_name: str = "studybuddy",
    loader_type: str = "pypdf",
) -> int:
    """
    Bir PDF'i chunk'layıp ChromaDB'ye ekler.

    Returns:
        Eklenen chunk sayısı.
    """
    chunks = load_and_chunk_pdf(pdf_path, loader_type=loader_type)
    embeddings = get_embedding_model()

    logger.info(f"ChromaDB'ye yazılıyor: {config.CHROMA_DIR}")
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    # chunk_id'leri unique document id olarak kullan (duplicate'leri önler)
    ids = [c.metadata["chunk_id"] for c in chunks]
    vectorstore.add_documents(documents=chunks, ids=ids)

    logger.info(f"  ✓ {len(chunks)} chunk eklendi")
    return len(chunks)


# ─── CLI Entry Point ───────────────────────────────────────────────────
def main() -> None:
    """Komut satırından: python -m studybuddy.ingestion <pdf_path>"""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Kullanım: python -m studybuddy.ingestion <pdf_path> [loader_type]")
        print("  loader_type: pypdf (default) veya pdfplumber")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).resolve()
    loader_type = sys.argv[2] if len(sys.argv) == 3 else "pypdf"
    config.validate_config()
    n = ingest_pdf(pdf_path, loader_type=loader_type)
    print(f"\n✓ {n} chunk başarıyla yüklendi.")


if __name__ == "__main__":
    main()