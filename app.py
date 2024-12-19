import streamlit as st
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
import os
import time
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

# Set page configuration at the very beginning
st.set_page_config(
    page_title="PostgreSQL Query Assistant", 
    page_icon="🤖", 
    layout="wide"
)

# Custom CSS for enhanced styling
def local_css():
    st.markdown("""
    <style>
    .main-container {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
    }
    .stApp {
        background-color: #ffffff;
    }
    .stChatInput {
        border-radius: 15px !important;
        border: 2px solid #3366cc !important;
    }
    .chat-header {
        background-color: #3366cc;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .chat-message {
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 10px;
    }
    .human-message {
        background-color: #e6f2ff;
        border-left: 4px solid #3366cc;
    }
    .ai-message {
        background-color: #f0f0f0;
        border-left: 4px solid #666666;
    }
    .sidebar .stTextInput > div > div > input {
        border-radius: 10px !important;
    }
    .sidebar {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def init_database(user: str, password: str, host: str, port: str, database: str, sslmode: str = None):
    """Initialize a connection to the PostgreSQL database."""
    try:
        db_uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        if sslmode:
            db_uri += f"?sslmode={sslmode}"
        
        db = SQLDatabase.from_uri(db_uri)
        return db
    except Exception as e:
        st.error(f"Unable to connect to the database: {e}")
        return None

def answer_sql(question: str, db, chat_history: list, llm):
    """Generate SQL answer based on the user's question and database content."""
    try:
        prompt = PromptTemplate(
            input_variables=['input', 'table_info', 'top_k'],
            template="""You are a PostgreSQL expert. Given an input question,
                        first create a syntactically correct PostgreSQL query to run,
                        then look at the results of the query and return the answer to the input question.
                        Unless the user specifies in the question a specific number of records to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL.
                        Wrap each column name in double quotes (") to denote them as delimited identifiers.
                        Only use the following tables:\n{table_info}\n\nQuestion: {input}')"""
        )

        db_chain = SQLDatabaseChain(
            llm=llm, 
            database=db, 
            top_k=100, 
            verbose=True, 
            use_query_checker=True, 
            prompt=prompt, 
            return_intermediate_steps=True
        )

        with get_openai_callback() as cb:
            response = db_chain.invoke({
                "query": question,
                "chat_history": chat_history,
            })["result"]

            # Optional: Log token usage (you can remove this or add logging as needed)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")

        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Sorry, I couldn't process your request."

def main():
    # Apply custom CSS
    local_css()

    # Main container
    with st.container():
        # Header
        st.markdown("<div class='chat-header'><h1 style='text-align: center;'>🤖 PostgreSQL Query Assistant</h1></div>", unsafe_allow_html=True)

        # Sidebar for connection
        with st.sidebar:
            st.image("https://www.postgresql.org/media/img/about/press/elephant.png", use_container_width=True)
            st.header("Database Connection")
            
            # Connection details
            with st.expander("Database Credentials", expanded=True):
                openai_api_key = st.text_input("OpenAI API Key", type="password", help="Required for natural language to SQL conversion")
                
                db_type = st.radio("Database Type", ("Local", "Cloud"))
                
                if db_type == "Local":
                    host = st.text_input("Host", value="localhost")
                    port = st.text_input("Port", value="5432")
                    user = st.text_input("Username", value="postgres")
                    password = st.text_input("Password", type="password")
                    database = st.text_input("Database Name", value="testing_3")
                    sslmode = None
                else:
                    host = st.text_input("Host (e.g., your-db-host.aws.com)")
                    port = st.text_input("Port", value="5432")
                    user = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    database = st.text_input("Database Name")
                    sslmode = st.selectbox("SSL Mode", ["require", "verify-ca", "verify-full", "disable"])

                connect_btn = st.button("🔌 Connect to Database")

        # Main chat area
        chat_container = st.container()

        # Initialize or load session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="👋 Hi there! I'm your PostgreSQL Query Assistant. Connect to your database and ask me anything!")
            ]
        
        if 'db_connected' not in st.session_state:
            st.session_state.db_connected = False

        # Connection handling
        if connect_btn:
            if not openai_api_key:
                st.error("Please provide an OpenAI API Key")
            else:
                os.environ["OPENAI_API_KEY"] = openai_api_key
                llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
                
                db = init_database(user, password, host, port, database, sslmode)
                
                if db:
                    st.session_state.db = db
                    st.session_state.llm = llm
                    st.session_state.db_connected = True
                    st.success("🎉 Successfully connected to the database!")

        # Display chat history
        with chat_container:
            for message in st.session_state.chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(f"<div class='chat-message ai-message'>{message.content}</div>", unsafe_allow_html=True)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(f"<div class='chat-message human-message'>{message.content}</div>", unsafe_allow_html=True)

        # Chat input and processing
        if st.session_state.db_connected:
            user_query = st.chat_input("Ask a question about your database...")
            
            if user_query:
                # Add user message to chat history
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                
                # Display user message
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"<div class='chat-message human-message'>{user_query}</div>", unsafe_allow_html=True)
                
                # Generate and display AI response
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Generating response..."):
                        response = answer_sql(
                            user_query, 
                            st.session_state.db, 
                            st.session_state.chat_history, 
                            st.session_state.llm
                        )
                    st.markdown(f"<div class='chat-message ai-message'>{response}</div>", unsafe_allow_html=True)
                
                # Add AI response to chat history
                st.session_state.chat_history.append(AIMessage(content=response))
        else:
            st.warning("Please connect to a database to start querying.")

if __name__ == "__main__":
    main()