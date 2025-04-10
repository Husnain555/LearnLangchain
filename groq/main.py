import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

load_dotenv()
groq_api_key = os.environ["LANGCHAIN_GROQ_API_KEY"]

if 'vectors' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    site_url = st.text_input("Please enter the site URL you want to interact with")

    if site_url:
        with st.spinner("Loading and processing documents..."):
            loader = WebBaseLoader(site_url)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=700)
            final_docs = text_splitter.split_documents(docs)
            st.session_state.vectors = FAISS.from_documents(final_docs, st.session_state.embeddings)
    else:
        st.info("Please enter a site URL above.")

st.title("Chat Groq Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")

prompt = ChatPromptTemplate.from_template('''
You are a helpful and precise assistant. Use ONLY the provided context to answer the question below.
- Extract ALL relevant information from the context.
- Do NOT make up any facts or guesses.
- Present the answer in a clear and structured way (e.g., lists, bullet points).
- If the context does not contain the answer, clearly state: "The context does not provide enough information."

<context>
{context}
</context>

Question: {input}
Answer:
''')


document_chain = create_stuff_documents_chain(llm, prompt)

if "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    user_prompt = st.text_input("Please input your prompt here")

    if user_prompt:
        start_time = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        st.write(response["answer"])
        st.caption(f"⏱️ Response time: {time.process_time() - start_time:.2f} seconds")

        with st.expander("Similarity- Search Results"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
else:
    st.info("👆 Please enter a website URL above and wait for it to finish processing before chatting.")
