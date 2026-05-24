# RAG Document Question Answering System

A Retrieval-Augmented Generation (RAG) based document question answering system built using LangChain and Streamlit.

## Features
- Upload PDF/documents
- Ask questions from uploaded documents
- Semantic search using vector embeddings
- LLM-generated responses
- Streamlit user interface

## Tech Stack
- Python
- LangChain
- Streamlit
- FAISS
- Groq API
- HuggingFace Embeddings

## Project Structure

```bash
rag_doc/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── ...
```

## Installation

1. Clone repository

```bash
git clone https://github.com/yourusername/GenAI.git
```

2. Navigate to project

```bash
cd GenAI/rag_doc
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Create `.env` file and add your API key

```env
GROQ_API_KEY=your_api_key
```

5. Run application

```bash
streamlit run app.py
```

## Author
Nidhi Patel