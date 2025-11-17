import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Annotated, Sequence, TypedDict, Dict

# ------------------------------------------------------------------
# 1️⃣  Imports – use the new 0.2+ locations
# ------------------------------------------------------------------
from langchain_groq import ChatGroq   
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from langchain.agents import Tool
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from splunk_tool import SplunkConfig, SplunkTool
load_dotenv()
os.environ["Groq_API_KEY"] = os.getenv("Groq_API_KEY_AgenticRAG")

#--------------------------
# 1. Create Retriever Tool
# --------------------------

# Load content from blog
docs = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

#above blogs data that was splited to chunks were embedded to Faiss vector store below 
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

# Creating a function to retrive data from my vector store 
def retriever_tool_func(**kwargs) -> str:
    # Accept either "query" or the positional "__arg1" key
    query = kwargs.get("query") or kwargs.get("__arg1")
    if not query:
        raise ValueError("No query provided to RAGRetriever")

    docs = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)

#Using above function as a tool 
retriever_tool = StructuredTool.from_function(
    func=retriever_tool_func,
    name="RAGRetriever",
    description="Use this tool to fetch relevant knowledge base info",
    arg_schema=Dict[str, str]  # {"query": str}
)
retriever_tool
print(retriever_tool.name)

# Wikipedia tool 
wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

# ------------------------------------------------------------------
# 2. Create the React agent node
# ----------------------------------------------------------------

os.environ["Groq_API_KEY"]=os.getenv("Groq_API_KEY_AgenticRAG")
# 2️⃣ Create the LLM instance
llm = ChatGroq(
    model_name="groq/llama-3.3-70b-versatile",   # the Groq model you want
    api_key=os.getenv("GROQ_API_KEY"),           # passes the key you just set
    temperature=0.2,                              # optional tuning knobs
    max_output_tokens=1024,                       # optional
)
tools = [retriever_tool, wiki_tool, splunk_tool]
react_node = create_react_agent(llm, tools)


# ------------------------------------------------------------------
# 3️⃣  Splunk Tool
# ------------------------------------------------------------------
splunk_client = SplunkTool(SplunkConfig())
# 3a.  Helper function – returns a **JSON‑string** so the LLM can parse it easily
def splunk_search_func(**kwargs) -> str:
    query = kwargs.get("query") or kwargs.get("__arg1")
    if not query:
        raise ValueError("No query provided to SplunkSearch")
    try:
        rows = splunk_client.search(query, timeout=60, earliest_time="-24h", latest_time="now")
    except Exception as exc:
        return f"{{\"error\": \"{str(exc)}\"}}"
    import json
    return json.dumps(rows, indent=2)
# 3b.  Create a StructuredTool (so the agent sees it as a “tool”)
splunk_tool = StructuredTool.from_function(
    func=splunk_search_func,
    name="SplunkSearch",
    description=(
        "Run a SPL query on your local Splunk instance (port 8000). "
        "Return results as a JSON array. Example: 'index=main | head 5'."
    ),
    arg_schema=Dict[str, str]
)
# Verify
print(f"🔌 Added tool: {splunk_tool.name}")
# ------------------------------------------------------------------
# 4. Build the graph (this is what FastAPI will call)
# ------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

builder = StateGraph(AgentState)
builder.add_node("react_agent", react_node)
builder.set_entry_point("react_agent")
builder.add_edge("react_agent", END)
graph = builder.compile()

# ✅ Export the graph object for FastAPI
__all__ = ["graph"]