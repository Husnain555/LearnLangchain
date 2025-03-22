from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama


app = FastAPI(
     title="LangChain Community API",
     description="LangChain Community API",
     version="1.0",
 )

add_routes(
    app,
    Ollama,
    path='openai'
)
llms = Ollama(model = "llama2")


prompt = ChatPromptTemplate.from_template("Write me a assay of 200 lines on this {topic}")
add_routes(
    app,
    prompt|llms,
    path="essay"
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port=8000)