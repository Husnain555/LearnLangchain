from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# === TOOLS SETUP ===
# Wikipedia Tool Setup
wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_api)

# Arxiv Tool Setup
arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_api)

# LangSmith Web Loader Setup for Geo News
web_loader = WebBaseLoader("https://www.geo.tv/")
docs = web_loader.load()

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Create Chroma vector store with Ollama Embeddings
db = Chroma.from_documents(text_splitter, OllamaEmbeddings())

# Set up the retriever
retriever = db.as_retriever()

# LangChain Tool setup for Geo News search
retrieval_tool = create_retriever_tool(
    retriever,
    name="Geo_News_Search",
    description=(
        "Use this tool to search detailed information about Geo News articles, "
        "including the latest news, trending topics, and other relevant content "
        "from the Geo TV website. Ideal when the user asks for news updates or "
        "specific articles from Geo TV."
    )
)
tools = [wiki, arxiv, retrieval_tool]

# === CUSTOM PROMPT ===
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intelligent research assistant with access to the following tools:\n\n"
     "1. **Wikipedia Search** – for general knowledge and widely known topics.\n"
     "2. **Arxiv Search** – for scientific papers, technical research, and academic content.\n"
     "3. **Geo News Search** – for the latest news, trending topics, and other relevant content from the Geo TV website. Ideal for news updates and specific articles related to Pakistan.\n\n"
     "Use these tools as needed to provide clear, detailed, and accurate answers. "
     "Always prioritize using the most relevant source based on the user's question. "
     "Cite retrieved information clearly if applicable, and keep your responses concise and helpful."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# === AGENT SETUP ===
llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)

# Creating the agent
agent = create_openai_tools_agent(
    tools=tools,
    llm=llm,
    prompt=prompt
)

# Now the agent is ready to be invoked
agents_x_executor = AgentExecutor(tools=tools, agent=agent, verbose=True)

# Initialize chat_history as an empty list or a suitable structure
chat_history = []

# When invoking the agent, include chat_history
agents_x_executor_invoke = agents_x_executor.invoke({
    "input": "Give me today latest Pakistani news",
    "chat_history": chat_history  # Include chat_history here
})
