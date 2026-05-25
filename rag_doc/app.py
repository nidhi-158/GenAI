import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import warnings

load_dotenv()

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


## load the GROQ 
#os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

groq_api_key = st.secrets["GROQ_API_KEY"]

st.title("RAG Document Q&A")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.loader = PyPDFDirectoryLoader("us_census")

        # Load PDFs
        st.session_state.docs = st.session_state.loader.load()

        # Debugging
        st.write(f"Total documents loaded: {len(st.session_state.docs)}")

        # Check if PDFs exist
        if len(st.session_state.docs) == 0:
            st.error("No PDF documents found in ./us_census folder")
            st.stop()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(
                st.session_state.docs[:20]
            )
        )

        st.write(f"Total chunks created: {len(st.session_state.final_documents)}")

        # Check chunks
        if len(st.session_state.final_documents) == 0:
            st.error("No document chunks created")
            st.stop()

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )


prompt1=st.text_input("Enter Your Question From Doduments")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

import time



if prompt1:

    if "vectors" not in st.session_state:
        st.warning("Please click on 'Documents Embedding' first.")
        st.stop()

    document_chain=create_stuff_documents_chain(llm,prompt)

    retriever=st.session_state.vectors.as_retriever()

    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()

    response=retrieval_chain.invoke({'input':prompt1})

    print("Response time :",time.process_time()-start)

    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
