# DocRAG – Offline PDF Question Answering using RAG and Ollama

DocRAG is an offline AI system that allows us to ask questions from PDF documents using Retrieval-Augmented Generation (RAG).
The system retrieves relevant content from the document and generates answers using a local language model running with Ollama.

Unlike traditional chatbots, DocRAG reduces hallucinations by grounding responses in the document context.

---

## Features

* Offline AI inference
* Question answering from PDF documents
* Retrieval-Augmented Generation (RAG)
* Local LLM using Ollama
* Vector similarity search
* Source citations with page numbers
* Highlighted paragraph in PDF
* Simple web interface using Streamlit

---

## Tech Stack

* Python
* LangChain
* FAISS vector database
* Sentence Transformers
* Ollama (Local LLM)
* PyMuPDF
* Streamlit

---

## System Architecture

PDF Document
↓
Text Chunking
↓
Embedding Model
↓
Vector Database (FAISS)
↓
User Question
↓
Similarity Search
↓
Retrieved Context
↓
Local LLM (Ollama)
↓
Answer + Source Citation

---

## Installation

Install dependencies

pip install -r requirements.txt

Install Ollama and download the model

ollama pull mistral

---

## Run the Application

streamlit run app.py

http://localhost:8501

Upload a PDF and start asking questions.

---

## Example Use Cases

* Research paper analysis
* Legal document search
* Study assistant for textbooks
* Corporate document search
* Knowledge management

---

## Author

Satyarth Singh
