import streamlit as st
import pandas as pd
import openai
from pinecone import Pinecone, ServerlessSpec
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
import asyncio
import time
import logging
from threading import Thread
import traceback
from slack_sdk.errors import SlackApiError
from datetime import datetime, timedelta
from functools import lru_cache
import os
from dotenv import load_dotenv
import nest_asyncio
import re
from langsmith import Client
from langchain.callbacks.tracers import LangSmithTracer
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import OpenAIEmbeddings

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configurations
DEFAULT_INDEX_NAME = "annual-leave"
DEFAULT_LANGCHAIN_PROJECT = "Leaves-Buddy-slack"
DEFAULT_LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

class LeaveBot:
    def __init__(self):
        self.initialize_state()
        self.setup_ui()
        self.setup_langsmith()
        self.setup_embeddings()
        
    def initialize_state(self):
        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False
        if 'logs' not in st.session_state:
            st.session_state.logs = ""
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        self.log_placeholder = st.empty()

    def setup_embeddings(self):
        self.embeddings = OpenAIEmbeddings(callbacks=[self.tracer])

    def setup_langsmith(self):
        # Set LangChain environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = DEFAULT_LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_PROJECT"] = DEFAULT_LANGCHAIN_PROJECT
        
        # Initialize tracer with default project
        self.tracer = LangSmithTracer(project_name=DEFAULT_LANGCHAIN_PROJECT)
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            callbacks=[self.tracer]
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("human", "{query}\n\nContext: {context}")
        ])
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            callbacks=[self.tracer]
        )

    @staticmethod
    def get_secret(key):
        """Get secret from Streamlit secrets or environment variables"""
        try:
            return st.secrets[key]
        except:
            return os.getenv(key)

    def get_system_prompt(self):
        """Get the detailed system prompt for the LLM"""
        today = datetime.now().strftime("%d-%m-%Y")
        return f"""You are LeaveBuddy, a precise leave management assistant. Today is {today}. 
        
Your primary tasks are:
1. Answer questions about employee leave schedules
2. Check who is on leave on specific dates
3. Provide leave information for specific employees
4. Help with leave planning and team availability

When responding:
1. Use only information from the provided context
2. Be precise with dates (use DD-MM-YYYY format)
3. Include the reason/festival for leaves
4. Clearly state if information is not available

For leave queries:
- Individual: "[Name] is on leave on [Date] for [Reason]"
- Date-based: List all employees on leave for that date
- Period queries: Group by date and list employees
- Team queries: Show overlapping leaves and availability

Always verify:
- Exact dates
- Employee names
- Leave reasons/festivals
- Data accuracy

Keep responses clear, concise, and accurate."""

    def update_log(self, message):
        """Update the log display with new messages"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.logs = f"[{timestamp}] {message}\n" + st.session_state.logs
        self.log_placeholder.text_area("Logs", st.session_state.logs, height=300)

    def create_structured_text(self, row):
        """Create a well-structured text representation of leave data"""
        return (
            f"{row['NAME']} has leave scheduled for {row['DATE']} ({row['DAY']}) "
            f"for {row['FESTIVALS']}. This leave is in {row['MONTH']} {row['YEAR']}. "
            f"The day is a {row['DAY']}. Festival/Event: {row['FESTIVALS']}. "
            f"Employee: {row['NAME']}. Full date: {row['DATE']}."
        )

    def create_embeddings(self, df):
        """Create embeddings for all leave records"""
        records = []
        for i, row in df.iterrows():
            text = self.create_structured_text(row)
            embedding = self.embeddings.embed_query(text)
            if embedding:
                records.append((
                    str(i),
                    embedding,
                    {
                        "text": text,
                        "date": row['DATE'],
                        "name": row['NAME'],
                        "day": row['DAY'],
                        "festival": row['FESTIVALS'],
                        "month": row['MONTH'],
                        "year": row['YEAR']
                    }
                ))
        return records

    def upload_to_pinecone(self, records):
        """Upload records to Pinecone index"""
        try:
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            index = pc.Index(index_name)
            index.upsert(vectors=records)
            return True, "Data uploaded to Pinecone successfully!"
        except Exception as e:
            logger.error(f"Error uploading to Pinecone: {e}")
            return False, f"Error uploading data to Pinecone: {str(e)}"

    async def query_pinecone(self, query):
        """Query Pinecone for relevant leave records"""
        try:
            index = pc.Index(index_name)
            query_embedding = self.embeddings.embed_query(query)
            
            results = index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            if results['matches']:
                matches = sorted(
                    results['matches'],
                    key=lambda x: datetime.strptime(x['metadata']['date'], '%d-%m-%Y')
                )
                context = " ".join([match['metadata']['text'] for match in matches])
                return context
            return None
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return None

    async def process_query(self, query):
        """Process a leave-related query and return a response"""
        try:
            logger.info(f"Processing query: {query}")
            
            context = await self.query_pinecone(query)
            
            if context:
                logger.info(f"Context found: {context[:200]}...")
                
                response = await self.chain.acall({
                    "query": query,
                    "context": context
                })
                
                return response['text']
            else:
                name_match = re.search(r"is (\w+)", query, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1)
                    today = datetime.now().strftime("%d-%m-%Y")
                    return f"Based on available information, {name} is present on {today}."
                else:
                    return "I couldn't find any relevant leave information. Please try rephrasing your question."
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return "I encountered an error while processing your query. Please try again later."

    def setup_slack_handlers(self):
        """Set up Slack event handlers"""
        @app.event("message")
        async def handle_message(event, say):
            try:
                text = event.get("text", "")
                channel = event.get("channel", "")
                
                # Send initial processing message
                processing_message = await app.client.chat_postMessage(
                    channel=channel,
                    text="Processing your request... :hourglass_flowing_sand:"
                )
                
                self.update_log(f"Received query: {text}")
                
                response = await self.process_query(text)
                
                if response:
                    try:
                        await app.client.chat_update(
                            channel=channel,
                            ts=processing_message['ts'],
                            text=response
                        )
                        
                        self.update_log(f"Responded: {response}")
                        
                    except SlackApiError as e:
                        logger.error(f"Error updating message: {e}")
                        await say(response)
                else:
                    await say("I'm sorry, I couldn't process your request. Please try again.")
                    
            except Exception as e:
                error_msg = f"Error in handle_message: {str(e)}"
                logger.error(error_msg)
                self.update_log(error_msg)
                await say("I'm sorry, I encountered an error. Please try again.")

    async def run_slack_bot(self):
        """Run the Slack bot"""
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()

    def start_bot_forever(self):
        """Start the bot in a continuous loop"""
        while True:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.run_slack_bot())
            except Exception as e:
                error_msg = f"Bot crashed: {e}"
                logger.error(error_msg)
                self.update_log(f"{error_msg} Restarting in 5 seconds...")
                time.sleep(5)

    def setup_ui(self):
        """Set up the Streamlit user interface"""
        st.title("Leave Buddy - Slack Bot")
        
        st.sidebar.header("Upload Leave Data")
        uploaded_file = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                required_columns = ['NAME', 'YEAR', 'MONTH', 'DATE', 'DAY', 'FESTIVALS']
                
                if not all(col in df.columns for col in required_columns):
                    st.sidebar.error("Excel file must contain all required columns: " + ", ".join(required_columns))
                    return
                
                df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%d-%m-%Y')
                
                st.sidebar.write("Uploaded Data Preview:")
                st.sidebar.dataframe(df.head())
                
                if st.sidebar.button("Process and Upload Data"):
                    with st.spinner("Processing and uploading data..."):
                        self.update_log("Creating embeddings...")
                        embeddings = self.create_embeddings(df)
                        
                        self.update_log("Uploading to Pinecone...")
                        success, message = self.upload_to_pinecone(embeddings)
                        
                        if success:
                            st.session_state['data_uploaded'] = True
                            st.session_state.last_update = datetime.now()
                            self.update_log("Data uploaded successfully!")
                            st.sidebar.success("Data processed and uploaded successfully!")
                        else:
                            self.update_log(f"Upload failed: {message}")
                            st.sidebar.error(message)
            
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                logger.error(error_msg)
                self.update_log(error_msg)
                st.sidebar.error(error_msg)

        st.header("Bot Controls")
        
        if st.session_state.last_update:
            st.info(f"Last data update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Bot", disabled=st.session_state.bot_running):
                st.session_state.bot_running = True
                st.write("Starting Slack bot...")
                self.setup_slack_handlers()
                thread = Thread(target=self.start_bot_forever, daemon=True)
                thread.start()
                st.success("Bot is running! You can now ask questions in Slack.")

        with col2:
            if st.session_state.bot_running and st.button("Stop Bot"):
                st.session_state.bot_running = False
                st.experimental_rerun()

def main():
    """Initialize and run the LeaveBot application"""
    try:
        # Initialize only the required API keys
        required_keys = [
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
            "SLACK_BOT_TOKEN",
            "SLACK_APP_TOKEN",
            "LANGCHAIN_API_KEY"
        ]
        
        # Check for required API keys
        missing_keys = [key for key in required_keys if not LeaveBot.get_secret(key)]
        if missing_keys:
            st.error(f"Missing required API keys: {', '.join(missing_keys)}")
            st.info("Please add the missing API keys to your .env file or Streamlit secrets.")
            return
            
        # Initialize global variables
        global pc, index_name, app
        
        # Set up Pinecone with default index name
        pc = Pinecone(api_key=LeaveBot.get_secret("PINECONE_API_KEY"))
        index_name = annual-leave
        
        # Set up Slack app
        app = AsyncApp(token=LeaveBot.get_secret("SLACK_BOT_TOKEN"))
        
        # Create and run the bot
        bot = LeaveBot()
        
    except Exception as e:
        st.error(f"Error initializing the application: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
