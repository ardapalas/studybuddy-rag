"""
StudyBuddy RAG — Streamlit UI.

Multi-session chat interface. Her kullanıcı kendi PDF'ini yükler,
session-scoped ChromaDB collection kullanılır.

Çalıştırmak için:
    streamlit run app.py
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

import streamlit as st

from studybuddy import config
from studybuddy.ingestion import ingest_pdf
from studybuddy.llm import generate_answer
from studybuddy.retrieval import retrieve

logging.basicConfig(level=logging.INFO)


# ─── Page Config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="StudyBuddy — Ders Asistanı",
    page_icon="📚",
    layout="centered",
)


# ─── Session Initialization ────────────────────────────────────────────
def init_session() -> None:
    """Session state'i ilk açılışta hazırla."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if "collection_name" not in st.session_state:
        st.session_state.collection_name = f"session_{st.session_state.session_id}"

    if "pdf_loaded" not in st.session_state:
        st.session_state.pdf_loaded = False

    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


init_session()


# ─── Sidebar: Upload + Info ────────────────────────────────────────────
with st.sidebar:
    st.title("📚 StudyBuddy")
    st.caption("Ders materyallerinden soru-cevap asistanı")

    st.divider()
    st.subheader("📤 PDF Yükle")

    uploaded = st.file_uploader(
        "Ders slaytı/notu (PDF)",
        type=["pdf"],
        help="Yüklediğin PDF'ten soru sorabilirsin.",
    )

    if uploaded is not None and not st.session_state.pdf_loaded:
        with st.spinner("PDF işleniyor... (embedding ~30 sn)"):
            # PDF'i session-scoped path'e kaydet
            session_upload_dir = config.UPLOADS_DIR / st.session_state.session_id
            session_upload_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = session_upload_dir / uploaded.name
            pdf_path.write_bytes(uploaded.getvalue())

            # Ingest et (session'a özel collection)
            try:
                n_chunks = ingest_pdf(
                    pdf_path,
                    collection_name=st.session_state.collection_name,
                )
                st.session_state.pdf_loaded = True
                st.session_state.pdf_name = uploaded.name
                st.session_state.n_chunks = n_chunks
                st.success(f"✓ {n_chunks} chunk yüklendi")
            except Exception as e:
                st.error(f"İngestion hatası: {e}")

    if st.session_state.pdf_loaded:
        st.info(
            f"📄 **{st.session_state.pdf_name}**\n\n"
            f"🧩 {st.session_state.n_chunks} chunk\n\n"
            f"🆔 Session: `{st.session_state.session_id}`"
        )

        if st.button("🔄 Yeni PDF yükle", use_container_width=True):
            # Session sıfırla
            for key in ["pdf_loaded", "pdf_name", "n_chunks", "messages"]:
                st.session_state.pop(key, None)
            # Yeni collection için yeni UUID
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.collection_name = f"session_{st.session_state.session_id}"
            st.rerun()

    st.divider()
    st.caption(
        "💡 **İpucu:** Temiz metinli slaytlar (PowerPoint export) "
        "en iyi sonucu verir. El yazısı/taranmış PDF'ler çalışmaz."
    )


# ─── Main Area: Chat ───────────────────────────────────────────────────
st.title("📚 Ders Asistanın")

if not st.session_state.pdf_loaded:
    st.info("👈 Başlamak için sol menüden bir PDF yükle.")
    st.stop()

# Geçmiş mesajları göster
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Kaynaklar"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}]** `{src['source']}` — sayfa {src['page']} "
                        f"(distance: {src['distance']:.3f})"
                    )
                    st.caption(src["text"][:300] + "...")

# Chat input
if prompt := st.chat_input("Sorunu yaz..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            chunks = retrieve(
                prompt,
                collection_name=st.session_state.collection_name,
            )
            result = generate_answer(prompt, chunks)

        st.markdown(result.answer)

        with st.expander("📚 Kaynaklar"):
            for i, src in enumerate(result.sources, 1):
                st.markdown(
                    f"**[{i}]** `{src.source}` — sayfa {src.page_display} "
                    f"(distance: {src.distance:.3f})"
                )
                st.caption(src.text[:300] + "...")

        # Geçmiş için kaydet (serializable form)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "sources": [
                {
                    "source": c.source,
                    "page": c.page_display,
                    "distance": c.distance,
                    "text": c.text,
                }
                for c in result.sources
            ],
        })