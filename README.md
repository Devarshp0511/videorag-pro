<div align="center">

# 🎬 VideoRAG Pro
### The Semantic Video Search & Analysis Engine

![RAG](https://img.shields.io/badge/Architecture-RAG-orange?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-cc2b5e?style=for-the-badge)
![Whisper](https://img.shields.io/badge/Audio_AI-Whisper-green?style=for-the-badge)
![Groq](https://img.shields.io/badge/Inference-Groq_LPU-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge)

**Transforming linear video content into an interactive Knowledge Graph.**
*Chat with YouTube videos, extract precise timestamps, and generate AI-powered summaries in seconds.*

[View Live Demo](https://videoragpro.streamlit.app/) · [Report Bug](https://github.com/Devarshp0511/videorag-pro/issues)

</div>

---

## 🔍 Executive Summary

**VideoRAG Pro** is an advanced Retrieval-Augmented Generation (RAG) system designed to solve the "Linear Consumption Problem" of video content. Traditional search engines index video metadata (titles, tags), but they cannot search inside the spoken content.

This application acts as a **Semantic Search Engine** for video. It transcribes audio, embeds the meaning into a high-dimensional vector space, and allows users to query the video using natural language (e.g., *"What is the main takeaway about neural networks?"*). The system retrieves the exact timestamp and uses a Large Language Model (LLM) to synthesize a direct answer.

![Application Demo](rag_demo.png)

---

## 🏗️ System Architecture

The pipeline consists of three distinct stages: **Ingestion**, **Indexing**, and **Retrieval**.

```mermaid
sequenceDiagram
    participant User
    participant Ingestion as Ingestion Engine
    participant VectorDB as ChromaDB
    participant LLM as Llama-3 (Groq)

    User->>Ingestion: Uploads Video / YouTube Link
    Ingestion->>Ingestion: 1. Audio Extraction (FFmpeg)
    Ingestion->>Ingestion: 2. ASR Transcription (Whisper)
    Ingestion->>Ingestion: 3. Semantic Chunking & Embedding (MiniLM)
    Ingestion->>VectorDB: 4. Store Vectors + Metadata {start_time}
    
    User->>VectorDB: Query: "Explain the concept of entropy"
    VectorDB->>VectorDB: 5. Cosine Similarity Search
    VectorDB->>LLM: 6. Retrieve Top Context Chunk
    LLM->>User: 7. Generate Answer + Timestamp

