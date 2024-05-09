import streamlit as st
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
from langchain_community.tools.tavily_search import TavilySearchResults
from itertools import count

def main():
    st.title("MA DPU 20-80 Docket Chatbot!")
    # st.sidebar.write("Sidebar Content")  # Add any sidebar content you need
    
    # Set up Langchain components
    openai_api_key = "..."
    llm = ChatOpenAI(openai_api_key=openai_api_key)

    x = FAISS.load_local("vector_store", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
    retriever = x.as_retriever(search_kwargs={"k": 10})
    retriever_tool = create_retriever_tool(
        retriever,
        "Massachusetts_2080_General_RAG",
        "This tool must be used to answer any specific questions about the Massachusetts 2080 proceeding a.k.a. the Future of Gas in Massachusetts, or the DPU's gas distribution proceeding.",
    )

    # Load Tavily API key and create search object
    os.environ["TAVILY_API_KEY"] = "..."
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    search_tool = TavilySearchResults(tavily_api_key=tavily_api_key)

    y = FAISS.load_local("pdf16_vector_store", OpenAIEmbeddings(openai_api_key=openai_api_key), allow_dangerous_deserialization=True)
    retriever_summary = y.as_retriever(search_kwargs={"k": 10})

    summary_retriever_tool = create_retriever_tool(
        retriever_summary,
        "Massachusetts_2080_Summary_RAG",
        "This tool must be used to answer any summary questions about the conclusion (aka anything related to the final order) \
            of the Massachusetts 2080 proceeding a.k.a. the Future of Gas in Massachusetts, or the DPU's gas distribution proceeding.",
    )

    # List of tools
    tools = [retriever_tool, search_tool, summary_retriever_tool]

    # Define prompt
    prompt =  ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You may not need to use tools for every query - the user may just want to chat!",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create Langchain agent
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    chat_history = ChatMessageHistory()

    conversational_agent_executor = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: chat_history,
        input_messages_key="input",
        output_messages_key="output",
        history_messages_key="chat_history",
    )

    user_input = st.text_input("You:", "")

    response = conversational_agent_executor.invoke(
        {
            "input": user_input,
        },
        {"configurable": {"session_id": "unused"}},
    )

    # If the user clicks the "Send" button
    if st.button("Send"):
        # Display the bot response
        st.write("Bot:", response["output"])
    


if __name__ == "__main__":
    main()


