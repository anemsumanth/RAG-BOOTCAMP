import os
from langgraph.prebuilt import create_react_agent
from langchain.agents import Tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import END
from langgraph.graph import StateGraph
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from langchain_core.tools import StructuredTool
from typing import Dict
"""
RAG‑Agent example that uses Groq LLaMA‑3.3‑70b, a local FAISS index and
Wikipedia as an optional tool.
"""

import os
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

# ------------------------------------------------------------------
# 1️⃣  Imports – use the new 0.2+ locations
# ------------------------------------------------------------------
from langchain.chat_models import init_chat_model
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.agents import Tool

# ------------------------------------------------------------------
# 2️⃣  Environment / credentials
# ------------------------------------------------------------------
from dotenv import load_dotenv

# --------------------------
# 1. Create Retriever Tool
# --------------------------

# Load content from blog
docs = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

#embedding = OpenAIEmbeddings()
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

# def retriever_tool_func(query: str) -> str:
#     print("📚 Using RAGRetriever tool")
#     docs = retriever.invoke(query)
#     return "\n".join([doc.page_content for doc in docs])

# retriever_tool=Tool(
#     name="RAGRetriever",
#     description="Use this tool to fetch relevant knowledge base info",
#     func=retriever_tool_func
# )


def retriever_tool_func(**kwargs) -> str:
    # Accept either "query" or the positional "__arg1" key
    query = kwargs.get("query") or kwargs.get("__arg1")
    if not query:
        raise ValueError("No query provided to RAGRetriever")

    docs = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in docs)

retriever_tool = StructuredTool.from_function(
    func=retriever_tool_func,
    name="RAGRetriever",
    description="Use this tool to fetch relevant knowledge base info",
    arg_schema=Dict[str, str]  # {"query": str}
)
retriever_tool
print(retriever_tool.name)
# Wikipedia tool
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wiki_tool


os.environ["Groq_API_KEY"]=os.getenv("Groq_API_KEY_AgenticRAG")
llm=init_chat_model("groq:llama-3.3-70b-versatile")
#llm=init_chat_model("openai:gpt-4o")
# 4️⃣  Quick test
if __name__ == "__main__":
    response = llm.invoke("Write a 1‑sentence summary of Groq's API.")
    print(response)
# ----------------------------
# 2. Define the Agent Node
# ----------------------------



tools = [retriever_tool, wiki_tool]

## create the native Langgraph react agent
react_node=create_react_agent(llm,tools)
react_node
# --------------------------
# 3. LangGraph Agent State
# --------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
# --------------------------
# 4. Build LangGraph Graph
# --------------------------

builder = StateGraph(AgentState)

builder.add_node("react_agent", react_node)
builder.set_entry_point("react_agent")
builder.add_edge("react_agent", END)

graph = builder.compile()
graph
# --------------------------
# 5. Run the ReAct Agent
# --------------------------

if __name__ == "__main__":
    user_query = "What is an agent loop and how does Wikipedia describe autonomous agents?"
    state = {"messages": [HumanMessage(content=user_query)]}
    result = graph.invoke(state)

    print("\n✅ Final Answer:\n", result["messages"][-1].content)