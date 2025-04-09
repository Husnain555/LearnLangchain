import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain  # FIXED typo: was `create_retrival_chain`
from langchain_community.vectorstores import FAISS
import time

load_dotenv()
groq_api_key = os.environ["LANGCHAIN_GROQ_API_KEY"]

if 'vector' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader(
        "https://coredirection.com/guides/home?_gl=1*1litgrk*_ga*MTQ4NDgzNDc2OC4xNzQ0MjIyOTE4*_ga_LVSSFZ40HP*MTc0NDIyMjk0Ni4xLjEuMTc0NDIyMjk1NC4wLjAuMA..")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.chunks_documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # FIXED: method call corrected to use split_documents() from TextSplitter instance
    st.session_state.final_document = st.session_state.chunks_documents.split_documents(st.session_state.docs)

    st.session_state.vectors = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)

st.title("Chat Groq Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-It")

# FIXED prompt: corrected typos and made instructions clearer and more accurate
prompt = ChatPromptTemplate.from_template('''
Use the provided context to answer the question as accurately as possible.
<context>{context}</context>
Question: {input}
''')

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

propmt = st.text_input("Please input your prompt here")

if propmt:
    start_time = time.process_time()  # FIXED: renamed 'time' var to 'start_time' to avoid conflict with imported module
    response = retrieval_chain.invoke({"input": propmt})  # FIXED: typo 'invock' -> 'invoke'
    print("Response_time:", time.process_time() - start_time)
    st.write(response)
