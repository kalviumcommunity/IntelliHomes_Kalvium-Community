<div align="center">

# IntelliHomes

### AI-powered Property Intelligence Platform

Ask questions about any property, understand complex legal documents, compare homes, and make confident real-estate decisions using Retrieval-Augmented Generation (RAG) and Large Language Models.

<!-- Badges -->

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![RAG](https://img.shields.io/badge/RAG-Powered-success)
![Status](https://img.shields.io/badge/Status-In_Development-orange)

</div>

---

## Why IntelliHomes?

Buying a home shouldn't require reading hundreds of pages of legal documents.

IntelliHomes transforms property documents into an intelligent conversational assistant that answers questions in plain English while citing the original source documents.

Instead of searching through PDFs, simply ask:

> "Who owns this property?"

> "Are there any legal disputes?"

> "Summarize the sale deed."

> "Compare these two apartments."

---

# Problem

Real estate decisions require analyzing multiple documents:

- Sale Deeds
- Encumbrance Certificates
- Tax Records
- Property Listings
- Locality Reports

These documents are:

- difficult to understand
- legally complex
- scattered across multiple sources
- time consuming to verify

IntelliHomes solves this by combining **AI + Document Retrieval**.

---

# Features

## AI Property Assistant

Chat naturally with your property documents.

```txt
You:
Is this property legally safe?

AI:
Based on the Sale Deed and Encumbrance Certificate,
there are no active legal claims...

(Source: SaleDeed.pdf Page 14)
```

---

## Document Intelligence

- Upload property PDFs
- AI document summarization
- Information extraction
- Source citations

---

## Smart Property Search

Search by

- Address
- City
- Locality
- Property Type
- Price
- Bedrooms

---

## Property Comparison

Compare multiple properties side-by-side.

- Amenities
- Price
- Legal Status
- Locality
- Documents

---

## Explain Legal Documents

No legal knowledge required.

Ask:

- What is an Encumbrance Certificate?
- Explain this clause.
- What does this legal term mean?

---

## Citation-Based Answers

Every response is backed by document citations.

No hallucinated answers.

---

# Architecture

```None
          User
            │
            ▼
      AI Chat Interface
            │
            ▼
     Retrieval Pipeline
            │
     ┌──────┴──────┐
     │             │
Vector Database   LLM
     │             │
     └──────┬──────┘
            │
      Property PDFs
```

---

# 🛠 Tech Stack

| Layer     | Technology      |
| --------- | --------------- |
| Backend   | Flask           |
| AI        | LLM             |
| Retrieval | RAG             |
| Documents | PDF Processing  |
| Search    | Semantic Search |
| Database  | _(To be added)_ |

---

# Project Structure

```text
backend/
frontend/
docs/
README.md
```

---

# Roadmap

- [ ] Property search
- [ ] PDF upload
- [ ] AI chatbot
- [ ] RAG pipeline
- [ ] Citation support
- [ ] Property comparison
- [ ] Authentication
- [ ] Deployment

---

# Future Scope

- Voice assistant
- OCR for scanned documents
- Regional language support
- Property valuation
- Mortgage assistant
- Mobile application

---

# Contributors

Made with ❤️ as part of the IntelliHomes project.

**Backend link:** https://intellihomes-kalvium-community.onrender.com/

## Run the Full App

Start the backend from the repository root:

```bash
cd backend
uv run python app.py
```

In a second terminal, start the React frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. In development, React calls the Flask API on port 5000. The working flows are property filtering, shortlist comparison, document upload and indexing, grounded questions, and source inspection.

## Deploy One Link

The root `render.yaml` configures a single Render web service with a persistent disk. It builds the React app, serves `frontend/dist` from Flask, runs the API with Gunicorn, and keeps accounts, uploads, embeddings, and SQLite metadata across restarts. To deploy:

1. Push this repository to GitHub.
2. In Render, choose **New > Blueprint** and select the repository.
3. Add `OPENAI_API_KEY` and any `OPENAI_BASE_URL` or model settings if live LLM responses are needed. Without a key, the project uses its deterministic retrieval/embedding fallback.
4. Deploy. Render provides one URL for both the website and its API.

The implemented account and data flows are: registration/login, authenticated saved shortlists, persisted property records, calculated comparison scores, upload indexing, document history, grounded document questions, and source citations.

Scanned-PDF OCR is enabled by default. Local setup requires the `tesseract-ocr` system package; Python dependencies are installed by `uv sync`. OCR is applied only to PDF pages with no text layer, then the recognized text follows the same chunking, embedding, indexing, and citation path as ordinary PDF text.

Still required for a full commercial product: a paid property-listing provider or your own verified listing import, object storage such as S3/Azure Blob for large documents, OCR for scanned PDFs, HTTPS/domain setup, transactional email, rate limiting, backups, and legal/security review. Those require external provider accounts, credentials, operational policies, and real data that are not present in this repository.

For another host, run `cd frontend && npm run build`, then serve `frontend/dist` and run `cd backend && uv run gunicorn -b 0.0.0.0:$PORT app:app` from the same service. Set `VITE_API_URL` only when the API is hosted separately.
