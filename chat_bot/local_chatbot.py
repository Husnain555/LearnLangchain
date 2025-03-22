from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful agent. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title("Local Chatbot using Ollama")
input_text = st.text_input("Enter your question:")

# LLM (Ensure `llama2` is installed in Ollama)
llm = Ollama(model="llama2")

# Output parser
output_parser = StrOutputParser()

# Create a chain
chain = prompt | llm | output_parser

# Process user input
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
