import os
import streamlit as st
from dotenv import load_dotenv
from langchain_agent import LangChainAgent

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="LangChain Agent Chatbot", layout="centered")
st.title("🤖 LangChain Agent Chatbot")
st.markdown("This chatbot matches all requirements from your screenshot. It can search the web, query a database, and read your documents.")

# --- API Key Management ---
with st.sidebar:
    st.header("Configuration")
    
    # Try to get keys from environment, fallback to user input
    groq_api_key = os.environ.get("GROQ_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    if not tavily_api_key:
        tavily_api_key = st.text_input("Tavily API Key", type="password")
        
    st.header("Tools")
    st.markdown("Try asking about:")
    st.markdown("- **Internet:** 'What's the weather in London?'")
    st.markdown("- **SQL:** 'Who has the highest salary in the company?'")
    st.markdown("- **Docs:** (Upload a PDF) 'Summarize this document.'")

# --- Agent Initialization (in Session State) ---
def init_agent(groq_key, tavily_key, model):
    """Initializes the agent and tools in session state."""
    if "agent_class" not in st.session_state:
        st.session_state.agent_class = LangChainAgent(
            groq_api_key=groq_key,
            tavily_api_key=tavily_key,
            groq_model=model
        )
    
    if "default_tools" not in st.session_state:
        st.session_state.default_tools = st.session_state.agent_class.get_default_tools()
    
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = st.session_state.agent_class.get_agent_executor(
            st.session_state.default_tools
        )
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# Check if keys are available
if not groq_api_key or not tavily_api_key:
    st.warning("Please enter your Groq and Tavily API keys in the sidebar to start.")
else:
    # Initialize the agent
    init_agent(groq_api_key, tavily_api_key, groq_model)

    # --- File Uploader (for Document Tool) ---
    uploaded_files = st.file_uploader(
        "Upload PDF(s) to add them as a tool for the agent", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Uploaded Files"):
            with st.spinner("Processing PDFs with LangChain..."):
                # 1. Create the new document tool
                doc_tool = st.session_state.agent_class.create_document_retriever_tool(uploaded_files)
                
                # 2. Get the full list of tools
                all_tools = st.session_state.default_tools + [doc_tool]
                
                # 3. Re-create the agent executor with the new tool
                st.session_state.agent_executor = st.session_state.agent_class.get_agent_executor(all_tools)
                
            st.success("Files processed! The agent can now answer questions about them.")
            st.warning("Note: Files are not saved. Re-upload is needed on page refresh.")

    # --- Display Chat History ---
    if "chat_history" in st.session_state:
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Bot:** {msg}")

    # --- Chat Input ---
    user_input = st.chat_input("Ask the agent anything...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        
        with st.spinner("Agent is thinking..."):
            try:
                # 1. Get current chat history from memory
                memory_history = st.session_state.agent_class.memory.load_memory_variables({})
                
                # 2. Invoke the agent
                response = st.session_state.agent_executor.invoke({
                    "input": user_input,
                    "chat_history": memory_history.get("chat_history", [])
                })
                
                # 3. Save interaction to memory
                st.session_state.agent_class.memory.save_context(
                    {"input": user_input},
                    {"output": response["output"]}
                )
                
                # 4. Add bot response to display history
                st.session_state.chat_history.append(("assistant", response["output"]))
                
                # 5. Refresh the UI
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.chat_history.append(("assistant", f"[Error] {e}"))
                st.rerun()