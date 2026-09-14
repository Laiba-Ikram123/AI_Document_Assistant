# 📄 AI Document Assistant

A simple Streamlit AI Document Assistant for uploading documents, searching their content, and asking questions using Grok.

## Features

- PDF, DOCX, TXT and Markdown upload
- Public Google Drive file/folder loading
- Document text extraction
- File name and PDF page metadata
- Overlapping text chunking
- Sentence Transformer embeddings
- FAISS semantic search
- Simple keyword search
- Hybrid semantic + keyword ranking
- Grok answers using retrieved document context only
- Retrieved sources shown after every answer
- Embeddings are created once when documents change
- Streamlit session state and caching
- Grok API key stored in Streamlit Secrets

## Files

```text
ai-document-assistant/
├── app.py
├── requirements.txt
└── README.md
```

## Pipeline

```text
Upload / Google Drive
        ↓
Text Extraction
        ↓
Chunking
        ↓
Embeddings
        ↓
FAISS Index
        ↓
User Question
        ↓
Question Embedding
        ↓
Semantic Search + Keyword Search
        ↓
Hybrid Ranking
        ↓
Relevant Chunks
        ↓
Grok
        ↓
Answer + Retrieved Sources
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For local development create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROK_API_KEY = "your_xai_api_key_here"
```

Never hardcode the API key in `app.py` or commit it to GitHub.

## Streamlit Cloud

Open your Streamlit app settings and add this to Secrets:

```toml
GROK_API_KEY = "your_xai_api_key_here"
```

## Run

```bash
streamlit run app.py
```

## Search

Semantic search uses FAISS and sentence embeddings.

Keyword search compares important words from the question with words in document chunks.

The final hybrid score is:

```text
70% Semantic Score + 30% Keyword Score
```

## Embedding Optimization

Document embeddings are generated only when the document collection changes.

The embeddings, chunks and FAISS index are stored in Streamlit session state.

When the user asks another question, the existing document embeddings are reused. Only the question is converted into an embedding.

## Supported Google Drive Files

- PDF
- DOCX
- TXT
- MD

Google Drive files/folders should be publicly accessible to the application.

## Page Metadata

PDF extraction preserves page numbers.

For DOCX, TXT and MD, page numbers are shown as unavailable because these formats do not reliably provide page information through the simple extraction pipeline.

## Future Improvements

- Excel and PowerPoint support
- OCR for scanned PDFs
- Persistent vector database
- Authentication
- Conversation history
- Advanced reranking
