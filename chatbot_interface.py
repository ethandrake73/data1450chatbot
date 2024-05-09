import os
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool

import chainlit as cl

openai_api_key = "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"
llm = ChatOpenAI(openai_api_key = openai_api_key)


# loading the vector embeddings as a retriever for the RAG
x = FAISS.load_local("vector_store", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)

retriever = x.as_retriever(search_kwargs={"k": 10})

retriever_tool = create_retriever_tool(
    retriever,
    "Massachusetts_2080_General_RAG",
    "This tool must be used to answer any specific questions about the Massachusetts 2080 proceeding a.k.a. the Future of Gas in Massachusetts, or the DPU's gas distribution proceeding.",
)

#TAVILY
from langchain_community.tools.tavily_search import TavilySearchResults

# load tavily API key 
os.environ["TAVILY_API_KEY"] = "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"

tavily_api_key = os.getenv("TAVILY_API_KEY")

# create search object 
search_tool = TavilySearchResults(tavily_api_key=tavily_api_key)

y = FAISS.load_local("pdf16_vector_store", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)

retriever_summary = y.as_retriever(search_kwargs={"k": 10})

# Summary RAG tool

summary_retriever_tool = create_retriever_tool(
    retriever_summary,
    "Massachusetts_2080_Summary_RAG",
    "This tool must be used to answer any summary questions about the conclusion (aka anything related to the final order) \
        of the Massachusetts 2080 proceeding a.k.a. the Future of Gas in Massachusetts, or the DPU's gas distribution proceeding.",
)

#List of tools
tools = [retriever_tool, search_tool, summary_retriever_tool]

# Get the prompt to use - you can modify this!
# prompt =  ChatPromptTemplate.from_messages(
# [
#     (
#         "system",
#         "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
#     ),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ]
# )

prompt =  ChatPromptTemplate.from_messages(
[
    (
        "system",
        "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]
)

# You need to set OPENAI_API_KEY environment variable or pass it as argument `api_key`.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@cl.on_chat_start
async def on_chat_start():
    prompt =  ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    #runnable = prompt | model | StrOutputParser()
    #cl.user_session.set("runnable", runnable)
    cl.user_session.set("llm_chain", agent_executor)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("llm_chain")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()