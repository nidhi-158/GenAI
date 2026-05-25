# GenAI
# RAG Document Q&A System

A Generative AI based Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Groq LLM.

This application allows users to upload and query PDF documents using natural language questions. The system retrieves relevant document chunks and generates accurate answers using an LLM.

---

## Features

- PDF document loading and processing
- Text chunking using LangChain
- Vector embeddings using HuggingFace
- FAISS vector database for semantic search
- Retrieval-Augmented Generation (RAG)
- Fast response generation using Groq LLM
- Streamlit interactive UI
- Document similarity search visualization

---

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API
- Sentence Transformers

---

## Project Structure

```bash
rag_doc/
│
├── app.py
├── requirements.txt
├── .env
├── docs/
│   └── PDF files
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/nidhi-158/GenAI.git
```

### Navigate to Project Folder

```bash
cd GenAI/rag_doc
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file inside the project folder and add:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## How It Works

1. Load PDF documents from the `docs` folder
2. Split documents into smaller chunks
3. Generate embeddings using HuggingFace model
4. Store embeddings in FAISS vector database
5. Retrieve relevant chunks based on user query
6. Generate contextual answers using Groq LLM

---

## Sample Workflow

- Click on **Documents Embedding**
- Wait for vector database creation
- Enter your question related to uploaded PDFs
- Get AI-generated answers with relevant document context

---

## Future Improvements

- Multiple file upload support
- Chat history memory
- Better UI design
- Support for DOCX and TXT files
- Deployment on Streamlit Cloud

---
