"""
LLM module — Groq + Llama 3.3 70B ile cevap üretimi.

Retrieval'ın sonuçlarını prompt'a gömer, kaynaklı Türkçe cevap üretir.
Prompt template buradadır — değiştirilmesi kolay olacak şekilde izole.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from langchain_groq import ChatGroq

from studybuddy import config
from studybuddy.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


# ─── Prompt Template ───────────────────────────────────────────────────
SYSTEM_PROMPT = """Sen bir ders asistanısın. Öğrenciye verilen ders \
materyallerinden soru cevaplıyorsun.

KURALLAR:
1. SADECE aşağıdaki CONTEXT'ten yararlanarak cevap ver.
2. CONTEXT'te yoksa "Bu bilgi verilen ders materyalinde yok." de, \
uydurma yapma.
3. Her SPESİFİK iddianın sonuna TEK kaynak numarası ekle: "X, Y'dir [2]." \
gibi. Toplu citation YAPMA: "[1][2][3][4]" YASAK.
4. Cevabını giriş/gelişme/sonuç yapısında YAPMA. Sadece soruyu yanıtla, \
gereksiz tekrar etme.
5. Cevabın Türkçe olsun, akademik ama anlaşılır bir dil kullan.
6. Kod/formül varsa aynen koru, kısa açıklama ekle.

ÖRNEK İYİ CEVAP:
"LCS algoritması iki dizinin en uzun ortak altdizisini bulur [1]. \
Özyinelemeli formül: x[i]=y[j] ise c[i,j]=c[i-1,j-1]+1, aksi halde \
max(c[i-1,j], c[i,j-1]) [2]."

ÖRNEK KÖTÜ CEVAP (yapma):
"Verilen kaynaklara göre [1][2][3] LCS algoritması... Sonuç olarak \
kaynaklar [1][2][3][4][5]'e göre bu algoritma..."
"""

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

SORU: {question}

CEVAP (kaynak göstererek):"""


# ─── Answer Container ──────────────────────────────────────────────────
@dataclass
class RAGAnswer:
    """Bir RAG cevabının tüm bileşenlerini tutar."""
    question: str                       # kullanıcının orijinal sorusu
    answer: str                         # LLM'in ürettiği cevap
    sources: list[RetrievedChunk]       # kullanılan kaynaklar (citation için)

    def format_sources(self) -> str:
        """README/UI için human-readable kaynak listesi."""
        lines = []
        for i, chunk in enumerate(self.sources, 1):
            lines.append(f"[{i}] {chunk.source}, sayfa {chunk.page_display}")
        return "\n".join(lines)


# ─── LLM Client Singleton ──────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Groq LLM client'i tek sefer init eder, cache'ler.

    ChatGroq altında httpx ile connection pool tutar —
    her çağrıda yeniden bağlantı kurmamak için singleton.
    """
    logger.info(f"LLM yükleniyor: {config.LLM_MODEL}")
    return ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        api_key=config.GROQ_API_KEY,
    )


# ─── Context Formatting ────────────────────────────────────────────────
def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Retrieved chunks'ı LLM'e verilecek context metnine dönüştür.

    Her chunk'a numara verir ([1], [2], ...), LLM bunları citation
    için kullanacak. Her chunk'tan sonra kaynak bilgisi de eklenir.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] (Kaynak: {chunk.source}, sayfa {chunk.page_display})\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


# ─── Main Generation Function ──────────────────────────────────────────
def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
) -> RAGAnswer:
    """
    Retrieved chunks + soru → LLM cevabı + kaynaklar.

    Args:
        question: Kullanıcı sorusu (ham, prefix'siz).
        chunks: retrieve() çıktısı.

    Returns:
        RAGAnswer dataclass (question, answer, sources).
    """
    if not chunks:
        return RAGAnswer(
            question=question,
            answer="İlgili bilgi bulunamadı. Soruyu farklı kelimelerle dene.",
            sources=[],
        )

    llm = get_llm()
    context = format_context(chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    logger.info(f"LLM çağrılıyor... (context: {len(context)} char)")
    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("user", user_prompt),
        ]
    )

    return RAGAnswer(
        question=question,
        answer=response.content,
        sources=chunks,
    )


# ─── CLI Entry Point ───────────────────────────────────────────────────
def main() -> None:
    """Standalone test: python -m studybuddy.llm "<soru>" """
    import sys

    from studybuddy.retrieval import retrieve

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print('Kullanım: python -m studybuddy.llm "<soru>"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    config.validate_config()

    chunks = retrieve(question)
    result = generate_answer(question, chunks)

    print(f"\n🔍 Soru: {result.question}\n")
    print("=" * 80)
    print(f"\n💬 Cevap:\n{result.answer}\n")
    print("=" * 80)
    print(f"\n📚 Kaynaklar:\n{result.format_sources()}")


if __name__ == "__main__":
    main()