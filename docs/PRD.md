# Product Requirements Document (PRD)

## IntelliHomes – AI-Powered Real Estate Assistant

**Version**: 1.0  
**Project Type**: Academic Class Project  
**Project Duration**: 6 Weeks  
**Team Size**: 3 Members

---

## Executive Summary

IntelliHomes is an AI-powered real estate assistant designed to help home buyers, renters,
property owners, and real estate agents make informed property decisions.

Traditional real estate platforms require users to manually verify information across multiple
documents such as sale deeds, encumbrance certificates, tax records, and locality reports.
This process is time-consuming and prone to human error.

IntelliHomes addresses this challenge by combining Retrieval-Augmented Generation (RAG)
with Large Language Models (LLMs) to provide trustworthy, document-grounded answers.
Users can search for properties, ask natural language questions, upload property-related
PDFs, and receive AI-generated responses with citations pointing to the relevant source
documents and page numbers.

The project focuses on improving information accessibility, transparency, and user
confidence while ensuring that AI responses remain explainable through document
references.

---

## Problem Statement

A real estate platform contains valuable information such as property listings, legal
documents, and locality reports. However, users must manually search through multiple
sources to answer questions about a property.

This creates several problems:

- Time-consuming document verification
- Difficulty understanding legal terminology

- Lack of a single trusted source of information
- Increased risk of overlooking important property details
- Poor user experience when evaluating multiple properties

There is a need for an intelligent assistant that can consolidate information from multiple
trusted documents and provide reliable, cited responses.

---

## Goals

### Primary Goals

- Build an AI-powered assistant for property-related queries.
- Enable users to ask questions in natural language.
- Provide answers grounded in uploaded documents.
- Display document citations with page or section references.
- Improve user satisfaction during property evaluation.
- Reduce manual effort involved in document analysis.

## Secondary Goals

- Compare multiple properties.
- Explain legal terminology in simple language.
- Summarize lengthy legal documents.
- Answer general real-estate questions.

---

## Non-Goals

The following features are intentionally out of scope for this project:

- Property purchasing or booking
- Online payments
- User authentication and authorization
- Recommendation engine
- Mortgage or loan approval
- Property price prediction
- Multi-language support
- Mobile application

---

## Target Users

**Home Buyers**

Need trustworthy information before purchasing a property.

**Goals**

- Verify legal safety
- Compare properties
- Understand documents
- Learn about the locality

---

**Renters**

Need quick insights about rental properties.

**Goals**

- Understand locality
- Compare options
- Ask general property-related questions

---

**Property Owners**

Need an easy way to organize and summarize property documents.

**Goals**

- Upload PDFs
- View summaries
- Retrieve information quickly

---

**Real Estate Agents**

Need faster access to property information while assisting clients.

**Goals**

- Retrieve property information quickly
- Explain legal documents
- Compare multiple properties

---

## User Stories

**Property Search**

- As a buyer, I want to search properties using multiple filters so that I can find suitable properties quickly.

**Property Details**

- As a user, I want to view all documents related to a property in one place.

**AI Chat**

- As a user, I want to ask questions in natural language instead of reading every document.

**Document Upload**

- As a property owner, I want to upload PDFs for AI analysis.

**Property Comparison**

- As a buyer, I want to compare two properties before making a decision.

**Legal Assistance**

- As a buyer, I want legal terms explained in simple language.

**Citation**

- As a user, I want every AI answer to include document citations with page references.

**Conversation History**

- As a user, I want previous conversations to be saved during my session for future reference.

---

## Functional Requirements

**Property Search**

The system shall allow users to:

- Search by property name
- Search by address
- Search by city
- Search by locality
- Search by pincode
- Apply filters (price, property type, bedrooms, etc.)

**Property Details**

The system shall display:

- Property information
- Property listing details
- Uploaded documents
- Locality information

**AI Chat Assistant**

The assistant shall answer questions regarding:

- Property listings
- Sale deeds
- Encumbrance certificates
- Building approvals
- Tax records
- Government records
- Locality reports
- Crime statistics
- Schools nearby
- General real estate laws
- Property buying process
- Legal terminology
- Investment-related insights

**Document Upload**

The system shall:

- Accept PDF documents only
- Process uploaded PDFs
- Extract text from PDFs
- Index extracted information for retrieval

**Document Summarization**

The AI shall generate summaries of uploaded documents.

**Property Comparison**

The assistant shall compare two selected properties based on available information.

**Citation Support**

Every AI-generated response shall include:

- Source document name
- Page number or section reference

**Saved Conversations**

The application shall maintain conversation history during the user's session.

---

## Non-Functional Requirements

**Performance**

- AI responses should be generated within a reasonable response time for a class project.

- Property searches should return results promptly.

**Reliability**

- The assistant should avoid generating answers without supporting evidence whenever possible.

**Explainability**

- Every factual answer should reference the supporting document.

**Scalability**

- The architecture should support adding additional data sources and larger document collections in the future.

**Cost**

- The entire solution should use free and open-source software.

**Deployment**

- The system should be deployable on commonly available hosting platforms.

---

## User Flow

**Property Search Flow**

```None

Home Page

      │

Search Property

      │

Select Property

      │

Property Details Page

      │

Ask AI Assistant

      │

Retrieve Relevant Chunks

      │

Generate AI Response

      │

Display Citations

      │

Continue Conversation
```

**Document Upload Flow**

```None

Upload PDF

      │

Extract Text

      │

Chunk Document

      │

Generate Embeddings

      │

Store in ChromaDB

      │

Ready for Retrieval
```

## System Overview

**Architecture**

```None

                     User

                      │

              Next.js Frontend

                      │

                FastAPI Backend

        ┌─────────────┴──────────────┐

        │                            │

Property Search AI Chat API

        │                            │

        │                    LangChain RAG

        │                            |

        │                   Chroma Vector DB

        │                            │

        │                  Retrieved Chunks

        │                            │

        │                 Llama 3.1 via Ollama

        │                            │

        └───────────────┬────────────┘

                        │

                  AI Response

                + Source Citations
```

## Technology Stack

| Component       | Technology                                                         |
| --------------- | ------------------------------------------------------------------ |
| Frontend        | Next.js (App Router)                                               |
| Backend         | FastAPI                                                            |
| LLM             | Llama 3.1 8B Instruct (or a smaller compatible version) via Ollama |
| Embeddings      | BAAI/bge-small-en-v1.5                                             |
| Vector Database | ChromaDB                                                           |
| Retrieval       | LangChain + Chroma                                                 |
| PDF Parsing     | PyMuPDF                                                            |
| Chunking        | Recursive Text Splitter                                            |
| Database        | SQLite (PostgreSQL optional for future scaling)                    |

---

## Acceptance Criteria

The project will be considered successful if users can:

- Search properties using multiple search methods.
- View detailed property information.
- Upload PDF documents.
- Ask natural language questions.
- Receive AI-generated answers grounded in uploaded documents.
- View document citations with page or section references.
- Compare two properties.
- Summarize uploaded documents.
- Ask general real-estate questions.
- View saved conversation history during their session.

## Success Metrics

**User Satisfaction**

- Users can successfully obtain useful answers without manually searching multiple documents.

**Query Success Rate**

- Percentage of user questions answered with relevant, cited information.

**Citation Coverage**

- Percentage of AI responses that include valid document citations.

**Property Comparison Success**

- Users can successfully compare properties using AI-generated summaries.

---

## Risks

| Risk                            | Mitigation                                                                         |
| ------------------------------- | ---------------------------------------------------------------------------------- |
| Hallucinated AI responses       | Restrict answers to retrieved documents where applicable and provide citations.    |
| Poor PDF quality                | Recommend clear, text-based PDFs; scanned PDFs may require OCR in future versions. |
| Limited hardware for local LLMs | Use a smaller Llama 3.1 model if necessary.                                        |
| Limited dataset                 | Begin with a curated sample dataset for development and testing.                   |
| Retrieval inaccuracies          | Tune chunk size, overlap, and embedding parameters during development.             |

---

## Milestones (6-Week Plan)

| Week   | Deliverables                                                                                                                          |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| Week 1 | Finalize requirements, prepare datasets, set up repository, implement PDF ingestion and text extraction.                              |
| Week 2 | Build chunking pipeline, generate embeddings, integrate ChromaDB, and validate document indexing.                                     |
| Week 3 | Develop the RAG retrieval pipeline, integrate LangChain, connect the local Llama model via Ollama, and implement citation generation. |
| Week 4 | Develop the Next.js frontend with property search, property details, AI chat interface, and document viewer.                          |
| Week 5 | Implement property comparison, document summarization, session-based conversation history, and end-to-end integration.                |
| Week 6 | Conduct testing, fix bugs, optimize retrieval quality and performance, prepare documentation, and deploy the application.             |
