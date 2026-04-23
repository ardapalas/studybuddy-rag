"""
Retrieval test scripti.
ChromaDB'deki embedding'lerin doğru çalıştığını görsel olarak doğrular.
"""
from langchain_chroma import Chroma

from studybuddy import config
from studybuddy.ingestion import get_embedding_model


def main() -> None:
    embeddings = get_embedding_model()
    vectorstore = Chroma(
        collection_name="studybuddy",
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    # DERSINE GÖRE DEĞİŞTİR — soru başında "query: " olmalı (E5 prefix)
    query = "query: LCS algoritması nasıl çalışır?"

    results = vectorstore.similarity_search_with_score(query, k=3)

    print(f"\n🔍 Sorgu: {query}\n")
    print(f"📚 Toplam chunk: {vectorstore._collection.count()}")
    print("=" * 80)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Match {i} (distance: {score:.4f}) ---")
        print(f"📄 Source: {doc.metadata.get('source')}")
        print(f"📃 Page: {doc.metadata.get('page')}")
        print(f"🆔 Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"📝 Text preview:")
        # passage: prefix'ini kaldırarak göster
        text = doc.page_content.replace("passage: ", "", 1)
        print(f"   {text[:300]}...")


if __name__ == "__main__":
    main()