from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langserve import add_routes
import uvicorn
from langchain_community.llms import Ollama

app = FastAPI(
    title="LangChain Community API",
    description="LangChain Community API",
    version="1.0",
)

# Define the model
llm = Ollama(model="llama2")

# Define the prompt template
prompt = ChatPromptTemplate.from_template("Write an essay of 200 lines about {topic}")

# Create a chain
essay_chain = LLMChain(prompt=prompt, llm=llm)

# Add the route
add_routes(app, essay_chain, path="/essay")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
