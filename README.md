# Vince_Agent - RAG Pipeline for International Students in Finland

A Retrieval-Augmented Generation (RAG) system designed to assist international students in Finland by providing accurate, source-cited answers to questions about student life, residence permits, housing, and other related topics.

## Table of Contents

- [Overview]
- [Architecture]
- [Features]
- [Prerequisites]
- [Installation]
- [Project Structure]
- [Usage]
- [API]
- [Known Limitations]
- [Future Improvements]
- [License]

---

## Overview

The Vince Agent is a locally-hosted RAG system that: 

1. **Retrieves** relevant information from a curated corpus of documents about Finland
2. **Augments** user queries with this contextual information
3. **Generates** accurate, source-cited responses using a local LLM
**
The system integrates with a mobile application, providing reliable, multilingual responses to user questions.**
---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VINCE AGENT PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    DATA      │    │  PROCESSING  │    │  EMBEDDING   │       │
│  │  COLLECTION  │───▶│    LAYER     │───▶│    LAYER     │       │
│  │    LAYER     │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│        │                    │                    │               │
│        ▼                    ▼                    ▼               │
│   ┌─────────┐        ┌───────────┐        ┌───────────┐         │
│   │ PDF     │        │ clean_    │        │ ChromaDB  │         │
│   │ HTML    │        │ corpus.    │        │ Vector    │         │
│   │ Markdown│        │ json      │        │ Store     │         │
│   └─────────┘        └───────────┘        └───────────┘         │
│                            │                    │               │
│                            ▼                    │               │
│                      ┌───────────┐              │               │
│                      │ chunked_  │              │               │
│                      │ corpus.    │              │               │
│                      │ json      │              │               │
│                      └───────────┘              │               │
│                                                 │               │
│  ┌──────────────────────────────────────────────┴────────────┐  │
│  │                     SERVICE LAYER                          │  │
│  │  ┌─────────┐    ┌───────────┐    ┌─────────┐             │  │
│  │  │ FastAPI │───▶│ Retrieval │───▶│ Ollama  │             │  │
│  │  │ Endpoint│    │ (ChromaDB)│    │ (LLM)   │             │  │
│  │  └─────────┘    └───────────┘    └─────────┘             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│                    ┌───────────────┐                           │
│                    │ Mobile App    │                           │
│                    │ (Frontend)    │                           │
│                    └───────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-format Support** | Processes PDF, HTML, and Markdown files |
| **Multilingual** | Supports Finnish and English queries using BGE-M3 embeddings |
| **Source Citations** | All responses include `[SOURCE_N]` citations |
| **Metadata Filtering** | Filter by partner organization and topic |
| **Local Deployment** | Runs entirely on local hardware (no cloud dependencies) |
| **RESTful API** | FastAPI with automatic OpenAPI documentation |

---

## Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.10+ | Runtime |
| Ollama | Latest | Local LLM hosting |
| RAM | 16GB+ recommended | Embedding model + vector store |
| Storage | 5GB+ | Models and vector database |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ItisMyHub/vince_project.git
cd vince-agent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama and Download Model

```bash
ollama pull llama3.2
```
---

## Project Structure

```
vince-agent/
├── README.md                      
├── requirements. txt              
│
├── data/
│   └── raw/                        
│       ├── *.pdf                   
│       ├── *. md                   
│       └── *.html                  
│
├── output/
│   ├── clean_corpus.json           
│   └── chunked_corpus.json         
│
├── chroma_db/                      
│
├── scripts/
│   ├── data_cleaner.py             
│   ├── corpus_chunker.py           
│   └── vector_store builder.py     
│
└── api_service. py                 
```

---

## Usage

### 1. Data Cleaning
```bash
python scripts/data_cleaner.py
```

**Input:** `raw_data/` (PDF, HTML, Markdown files)  
**Output:** clean_corpus.json`
Removes HTML noise, extracts PDF text, detects duplicates, and filters short content

### 2. Chunking

Split cleaned documents into semantic chunks:

```bash
python scripts/corpus_chunker.py
```

**Input:** `clean_corpus.json`  
**Output:** `chunked_corpus.json`

Configuration:
- Chunk size: 1,500 characters, Overlap: 200 characters
- Recursive character splitting (paragraph → sentence → word)

### 3. Embedding

```bash
python scripts/vector_store_builder.py
```
Generates vectors using BAAI/bge-m3 (multilingual, 1024D)
Stores them in chroma_db/ for retrieval

### 4. Running the API

Start Ollama(keep running):
```bash
ollama serve
```

```
Start FastAPI service:

```bash
python api_service.py
```

The API will be available at: 
- **Base URL:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

Known Limitations:

CPU-only inference takes 15–30s
Answers follow the language of source documents
No automatic translation across languages

Future Improvements:

GPU acceleration for real-time responses
Automatic language translation
Distributed vector storage for large corpora


