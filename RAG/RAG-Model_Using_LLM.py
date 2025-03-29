from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

# Load documents
loader = TextLoader("speech.txt")
docs = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
document = text_splitter.split_documents(docs)

# Vectorstore with Ollama Embeddings
db = Chroma.from_documents(document[:20], OllamaEmbeddings(model="llama2"))

# Setup LLM
llm = Ollama(model="llama2")

# Setup Prompt Template
prompt = ChatPromptTemplate.from_template("""
answer the following question based on the provided context. Think carefully before answering, it's very high-quality data.
<context>{context}</context>
Question: {input}
""")

# Create chains
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Query the chain
def data_question(question):
    return retrieval_chain.invoke({"input": question})

# Streamlit UI
st.title("RAG Question Answering based on Text file")
question = st.text_input("Please enter your question:")

if question:
    response = data_question(question)
    st.write(response["answer"])
