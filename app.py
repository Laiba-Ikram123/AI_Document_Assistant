import os
import re
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import faiss
import gdown

from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from openai import OpenAI


st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")

st.title("📄 AI Document Assistant")
st.write("Upload documents or load them from Google Drive, then ask questions using the document content.")


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROK_MODEL = "grok-4.6"


if "documents" not in st.session_state:
    st.session_state.documents = {}
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


def extract_pdf(file_bytes, file_name):
    pages = []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        reader = PdfReader(temp_path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "file_name": file_name,
                    "page": page_number,
                    "text": text.strip()
                })
    finally:
        os.remove(temp_path)

    return pages


def extract_docx(file_bytes, file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        document = Document(temp_path)
        paragraphs = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if text:
                paragraphs.append(text)

        text = "\n".join(paragraphs)
    finally:
        os.remove(temp_path)

    if not text.strip():
        return []

    return [{
        "file_name": file_name,
        "page": None,
        "text": text
    }]


def extract_text_file(file_bytes, file_name):
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    return [{
        "file_name": file_name,
        "page": None,
        "text": text.strip()
    }]


def extract_document(file_bytes, file_name):
    extension = Path(file_name).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, file_name)
    if extension == ".docx":
        return extract_docx(file_bytes, file_name)
    if extension in [".txt", ".md"]:
        return extract_text_file(file_bytes, file_name)

    return []


def create_chunks(extracted_sections):
    chunks = []

    for section in extracted_sections:
        text = section["text"]
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "file_name": section["file_name"],
                    "page": section["page"]
                })

            if end >= len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


def calculate_document_hash(documents):
    hasher = hashlib.sha256()

    for file_name in sorted(documents.keys()):
        hasher.update(file_name.encode("utf-8"))
        hasher.update(documents[file_name])

    return hasher.hexdigest()


def build_vector_index(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embeddings, index


def process_documents(documents):
    extracted_sections = []

    for file_name, file_bytes in documents.items():
        extracted_sections.extend(
            extract_document(file_bytes, file_name)
        )

    chunks = create_chunks(extracted_sections)

    if not chunks:
        return [], None, None

    embeddings, index = build_vector_index(chunks)

    return chunks, embeddings, index


def important_words(text):
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower())

    stop_words = {
        "what", "which", "when", "where", "who", "how",
        "does", "this", "that", "with", "from", "about",
        "into", "have", "has", "are", "the", "and", "for", "can"
    }

    return {word for word in words if word not in stop_words}


def keyword_score(question, chunk_text):
    question_words = important_words(question)

    chunk_words = set(
        re.findall(r"\b[a-zA-Z0-9]{3,}\b", chunk_text.lower())
    )

    if not question_words:
        return 0.0

    matches = question_words.intersection(chunk_words)
    return len(matches) / len(question_words)


def hybrid_search(question, top_k=TOP_K):
    chunks = st.session_state.chunks
    index = st.session_state.faiss_index

    if not chunks or index is None:
        return []

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    search_k = min(max(top_k * 3, 10), len(chunks))

    semantic_scores, semantic_ids = index.search(
        question_embedding,
        search_k
    )

    candidates = {}

    for score, chunk_id in zip(semantic_scores[0], semantic_ids[0]):
        if chunk_id == -1:
            continue

        candidates[int(chunk_id)] = {
            "semantic_score": float(score),
            "keyword_score": 0.0
        }

    for chunk_id, chunk in enumerate(chunks):
        score = keyword_score(question, chunk["text"])

        if chunk_id in candidates:
            candidates[chunk_id]["keyword_score"] = score
        elif score > 0:
            candidates[chunk_id] = {
                "semantic_score": 0.0,
                "keyword_score": score
            }

    results = []

    for chunk_id, scores in candidates.items():
        hybrid_score = (
            0.70 * scores["semantic_score"]
            + 0.30 * scores["keyword_score"]
        )

        result = chunks[chunk_id].copy()
        result["semantic_score"] = scores["semantic_score"]
        result["keyword_score"] = scores["keyword_score"]
        result["hybrid_score"] = hybrid_score

        results.append(result)

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return results[:top_k]


def get_grok_client():
    api_key = st.secrets.get("GROK_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )


def ask_grok(question, retrieved_chunks):
    client = get_grok_client()

    if client is None:
        return "Grok API key is not configured. Add GROK_API_KEY to Streamlit secrets."

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        page_text = (
            f"Page {chunk['page']}"
            if chunk["page"] is not None
            else "Page not available"
        )

        context_parts.append(
            f"SOURCE {number}\n"
            f"File: {chunk['file_name']}\n"
            f"{page_text}\n\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI document assistant.

Answer the user's question ONLY using the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the provided context, clearly say that the information is not available in the documents.
4. Give a clear and concise answer.
5. When useful, mention the relevant document name.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}
"""

    response = client.responses.create(
        model=GROK_MODEL,
        input=prompt
    )

    return response.output_text


def load_from_google_drive(url):
    temp_directory = tempfile.mkdtemp()
    downloaded_files = {}

    try:
        if "/folders/" in url:
            downloaded = gdown.download_folder(
                url,
                output=temp_directory,
                quiet=True,
                use_cookies=False
            )

            if downloaded:
                for file_path in downloaded:
                    path = Path(file_path)

                    if path.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]:
                        downloaded_files[path.name] = path.read_bytes()
        else:
            output_path = os.path.join(temp_directory, "drive_file")

           else:
    output_path = os.path.join(temp_directory, "drive_file")

    downloaded_path = gdown.download(
        url,
        output_path,
        quiet=True
    )

    if downloaded_path:
        path = Path(downloaded_path)

        if path.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]:
            downloaded_files[path.name] = path.read_bytes()
           
        return downloaded_files

    except Exception as error:
        st.error(f"Google Drive loading failed: {error}")
        return {}


st.sidebar.header("📚 Add Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)

drive_url = st.sidebar.text_input(
    "Google Drive file/folder link",
    placeholder="Paste a public Google Drive link"
)

if st.sidebar.button("Load Google Drive"):
    if drive_url.strip():
        with st.spinner("Loading Google Drive files..."):
            drive_files = load_from_google_drive(drive_url.strip())

        if drive_files:
            st.session_state.documents.update(drive_files)
            st.session_state.processed_hash = None
            st.success(f"Loaded {len(drive_files)} supported file(s) from Google Drive.")
    else:
        st.warning("Please paste a Google Drive link.")


if uploaded_files:
    for uploaded_file in uploaded_files:
        st.session_state.documents[uploaded_file.name] = uploaded_file.getvalue()

    st.session_state.processed_hash = None


if st.session_state.documents:
    current_hash = calculate_document_hash(st.session_state.documents)

    if st.session_state.processed_hash != current_hash:
        with st.spinner("Extracting, chunking and embedding documents..."):
            chunks, embeddings, index = process_documents(
                st.session_state.documents
            )

            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.faiss_index = index
            st.session_state.processed_hash = current_hash

        st.success("Documents processed successfully.")


if st.session_state.documents:
    st.header("📋 Document Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Documents", len(st.session_state.documents))

    with col2:
        st.metric("Created Chunks", len(st.session_state.chunks))

    for file_name, file_bytes in st.session_state.documents.items():
        extension = Path(file_name).suffix.upper()

        with st.expander(f"📄 {file_name}"):
            st.write(f"**File type:** {extension}")
            st.write(f"**File size:** {len(file_bytes) / 1024:.1f} KB")


if st.session_state.chunks:
    with st.expander("🔹 View Created Chunks"):
        for number, chunk in enumerate(st.session_state.chunks[:20], start=1):
            page = chunk["page"] if chunk["page"] is not None else "N/A"

            st.markdown(
                f"**Chunk {number}** — "
                f"{chunk['file_name']} — Page: {page}"
            )

            st.write(chunk["text"][:500])
            st.divider()


st.header("💬 Ask Your Documents")

question = st.text_input(
    "Ask a question about your documents",
    placeholder="Example: What is the main objective of this document?"
)

if st.button("Ask Grok"):
    if not st.session_state.chunks:
        st.warning("Please upload or load a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents..."):
            retrieved_chunks = hybrid_search(question)

        if not retrieved_chunks:
            st.warning("No relevant information was found.")
        else:
            with st.spinner("Generating answer with Grok..."):
                answer = ask_grok(question, retrieved_chunks)

            st.subheader("🤖 Answer")
            st.write(answer)

            st.subheader("📚 Retrieved Sources")

            for number, chunk in enumerate(retrieved_chunks, start=1):
                page = (
                    chunk["page"]
                    if chunk["page"] is not None
                    else "Not available"
                )

                with st.expander(
                    f"Source {number}: {chunk['file_name']} (Page: {page})"
                ):
                    st.write(f"**File:** {chunk['file_name']}")
                    st.write(f"**Page:** {page}")
                    st.write(f"**Hybrid Score:** {chunk['hybrid_score']:.3f}")
                    st.write(f"**Semantic Score:** {chunk['semantic_score']:.3f}")
                    st.write(f"**Keyword Score:** {chunk['keyword_score']:.3f}")
                    st.markdown("**Retrieved Text:**")
                    st.write(chunk["text"])
else:
    st.info("Upload a PDF, DOCX, TXT or MD file, or load documents from Google Drive to begin.")
