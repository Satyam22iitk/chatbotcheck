import pandas as pd
from io import BytesIO

# --- LLM ---
from langchain_groq import ChatGroq

# --- Agent & Memory ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

# --- Tool Imports ---
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchTool
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- Document RAG Tool Imports ---
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retriever_tool
from pypdf import PdfReader


class LangChainAgent:
    """
    This class builds a modular LangChain-powered agent that satisfies
    all the requirements from the user's screenshot.
    """

    def __init__(self, groq_api_key: str, tavily_api_key: str, groq_model: str):
        self.groq_api_key = groq_api_key
        self.tavily_api_key = tavily_api_key

        # 1. Initialize the LLM
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=groq_model,
            temperature=0
        )

        # 2. Initialize Embeddings (runs locally)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # 3. Initialize Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        # 4. Initialize Conversation Memory
        # "context-aware memory"
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,  # Remembers last 5 interactions
            return_messages=True
        )

    def _create_sql_tool(self) -> Tool:
        """
        Creates the "SQL queries" tool.
        We'll create a sample in-memory database.
        """
        # Create a sample dataset
        data = {
            'EmployeeID': [101, 102, 103, 104, 105],
            'Name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown', 'David Lee', 'Emily White'],
            'Department': ['Sales', 'Engineering', 'Marketing', 'Engineering', 'Sales'],
            'Salary': [70000, 95000, 62000, 105000, 72000]
        }
        df = pd.DataFrame(data)

        # Create a Pandas Agent. This agent can inspect and query the dataframe.
        # This is a simple way to fulfill the "SQL query" requirement.
        pandas_agent_executor = create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            verbose=True,
            handle_parsing_errors=True
        )

        # Wrap the pandas agent in a Tool
        return Tool(
            name="EmployeeDatabaseTool",
            func=pandas_agent_executor.invoke,
            description=(
                "Use this tool to answer questions about employees, departments, and salaries. "
                "Input should be a full question about the employee database."
            )
        )

    def _create_internet_search_tool(self) -> Tool:
        """
        Creates the "internet-connected responses" tool.
        """
        return TavilySearchTool(api_key=self.tavily_api_key)
    
    def create_document_retriever_tool(self, uploaded_files: list) -> Tool:
        """
        Creates the "document parsing" and "retrieval" tool.
        This is called *only* when files are uploaded.
        """
        print("Creating document retriever tool...")
        docs = []
        for file_like in uploaded_files:
            # Read PDF from in-memory file
            content = file_like.read()
            reader = PdfReader(BytesIO(content))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_like.name, "page": i + 1}
                    ))

        if not docs:
            # Return a dummy tool if no text is extracted
            return Tool(
                name="DocumentSearchTool",
                func=lambda x: "No documents have been uploaded or processed. Please ask the user to upload PDFs.",
                description="Use this tool to answer questions from uploaded PDF documents."
            )

        # Split, Embed, and Store in FAISS
        chunks = self.text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Create a retriever tool
        # This is a key LangChain module
        return create_retriever_tool(
            retriever=retriever,
            name="DocumentSearchTool",
            description="Use this tool to answer questions about the user-uploaded PDF documents."
        )

    def get_agent_executor(self, tools: list) -> AgentExecutor:
        """
        Builds the final AgentExecutor using the provided tools.
        This combines:
        - "optimized prompts"
        - "API-based tools"
        - "context-aware memory"
        """
        # 1. Create the "Optimized Prompt"
        # This prompt tells the agent how to behave
        system_prompt = """
        You are a helpful assistant.
        You have access to a suite of tools.
        Use the tools to answer the user's questions.
        If the answer is not in your tools, use your general knowledge.
        
        - For questions about employees, salaries, or departments, use 'EmployeeDatabaseTool'.
        - For current events, web searches, or "website interaction", use 'TavilySearchTool'.
        - For questions about the user's uploaded files, use 'DocumentSearchTool'.
        
        Be concise and helpful in your responses.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # 2. Create the Agent
        # This uses LangChain's `create_tool_calling_agent`
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        # 3. Create the Agent Executor
        # This is the final runnable "modular LangChain-powered chatbot"
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True, # Set to True to see agent's thoughts
            handle_parsing_errors=True
        )

        return agent_executor

    def get_default_tools(self) -> list:
        """
        Returns the list of tools that are always available.
        """
        sql_tool = self._create_sql_tool()
        search_tool = self._create_internet_search_tool()
        return [sql_tool, search_tool]