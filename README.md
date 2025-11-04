# 📚 DocuMind AI

> Multimodal Retrieval-Augmented Generation (RAG) System for Intelligent Document Understanding

## 🎯 Overview

DocuMind AI is an experimental multimodal RAG system designed to understand and answer questions about PDF and Word documents. This project explores cutting-edge techniques in semantic chunking, cross-modal embeddings, and context-aware language generation.

## 🔥 Key Features

- 📄 Multimodal Document Processing - Extract text and images from PDFs and Word documents
- 🖼️ Cross-modal Search - Query across text and images in a unified embedding space
- 🚀 Production-Ready Deployment - End-to-end deployment on Hugging Face Spaces
- ⚡ Real-time Q&A - Get intelligent answers from your documents instantly
- 🔬 Experimental Framework - Designed for rapid iteration and technique exploration

## 🌐 Live Demo

Try DocuMind AI on Hugging Face Spaces:
https://huggingface.co/spaces/Sakshi0104/documind-ai

## 🏗️ Architecture

PDF/Word Document
    ↓
Text + Image Extraction (PyMuPDF, python-docx)
    ↓
Semantic Chunking (LangChain - 500 chars, 50 overlap)
    ↓
Multimodal Embeddings (CLIP openai/clip-vit-base-patch32)
    ↓
Vector Store (FAISS - CPU Index)
    ↓
Semantic Search (Top-K=3 Retrieval)
    ↓
Context + Query → LLM (Groq - Llama 3.1 8B)
    ↓
Intelligent Answer

## 🧠 Technical Stack

Current Implementation:
- Chunking: Fixed-Size Character (500 chars, 50 overlap)
- Embeddings: CLIP ViT-Base-32 (512-dim vectors)
- Vector DB: FAISS (CPU-optimized)
- LLM: Llama 3.1 8B (Groq API)
- Framework: LangChain (RAG orchestration)
- UI: Gradio 4.44.0 (Web interface)

## 📦 Installation & Local Deployment

Prerequisites:
- Python 3.10+
- pip / conda
- Git

Setup Steps:

1. Clone Repository:
git clone https://github.com/Sakshiv0104/DocuMind-AI.git
cd DocuMind-AI

2. Create Virtual Environment:
python -m venv venv
source venv/bin/activate

3. Install Dependencies:
pip install -r requirements.txt

4. Set Environment Variables:
Create .env file with:
GROQ_API_KEY=your_groq_api_key_here

5. Run Locally:
python app.py
Access at http://localhost:7860

## 🚀 Hugging Face Spaces Deployment

Live Link: https://huggingface.co/spaces/Sakshi0104/documind-ai
Status: Running (Free Tier - CPU)
Cold Start: ~5-10 minutes (CLIP model loading)
Warm State: 30 seconds per query

## 🔧 Configuration

Edit config.py to customize:

CHUNK_SIZE = 500              # Characters per chunk
CHUNK_OVERLAP = 50            # Overlap between chunks
TOP_K = 3                     # Number of chunks to retrieve
MAX_PAGES_PER_PDF = 5         # Pages to process per PDF
EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
LLM_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.7
MAX_TOKENS = 1024

## 📊 Performance

Cold Start: 5-10 min
Warm Query: ~30 sec
Embedding Time: 2-5 sec
Retrieval Speed: <100ms

## 🎯 Vision

DocuMind AI represents my exploration of modern AI techniques in document understanding and retrieval-augmented generation. Through systematic experimentation with different chunking strategies, embedding models, and vector databases, I aim to build a robust, adaptable system that demonstrates best practices in multimodal AI.

This project showcases end-to-end deployment from local development through production deployment on Hugging Face Spaces. As I continue this journey, I aspire to contribute meaningfully to the open-source AI community and explore emerging techniques in semantic search, prompt engineering, and knowledge retrieval systems.

## 🔗 Quick Links

Live App: https://huggingface.co/spaces/Sakshi0104/documind-ai
GitHub: https://github.com/Sakshiv0104/DocuMind-AI

Status: In Active Development - Expect updates and experimental features!
