from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
import uvicorn
from langserve import add_routes

app = FastAPI(
    title="Api for voice chat model",
    description="A simple API for voice chat modal",
    version="1.0",
)
prompt = ChatPromptTemplate.from_template(
    "Give a {tip} on health sector",
)
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

chain = LLMChain(prompt=prompt, llm=llm, output_parser=output_parser)


add_routes(app, chain,path="/voice/agent")

uvicorn.run(app,host="localhost",port=8000)


