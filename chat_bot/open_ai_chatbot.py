from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import dotenv
import os
import streamlit as st

# Load environment variables
dotenv.load_dotenv()

# Set API key for OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Optional: Enable LangSmith tracing if needed
os.environ["LANGSMITH_TRACING"] = "true"

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Streamlit integration
st.title("Chat with ChatBot using OpenAI API")

# User input
input_text = st.text_input("Enter your question:")

# OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Output parser
output_parser = StrOutputParser()

# Create a chain
chain = prompt | llm | output_parser

# Process user input
if input_text:
    response = chain.invoke({"question": input_text})  # Fixed dictionary syntax
    st.write(response)
