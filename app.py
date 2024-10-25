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
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks import tracing_enabled
from langchain.callbacks.manager import CallbackManager

# Apply nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

# Initialize APIs
openai.api_key = get_secret("OPENAI_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
SLACK_BOT_TOKEN = get_secret("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = get_secret("SLACK_APP_TOKEN")

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = get_secret("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Leaves-Buddy"

# Initialize LangChain tracer
tracer = LangChainTracer(
    project_name="Leaves-Buddy"
)

# Initialize ChatOpenAI with optimized settings
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.1,
    streaming=False,
    request_timeout=10,
    callback_manager=CallbackManager([tracer])
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "annual-leave"

# Initialize Slack
app = AsyncApp(token=SLACK_BOT_TOKEN)

class LeaveBot:
    def __init__(self):
        self.initialize_state()
        self.setup_ui()
        self.setup_langchain()
        
    def initialize_state(self):
        if 'bot_running' not in st.session_state:
            st.session_state.bot_running = False
        if 'logs' not in st.session_state:
            st.session_state.logs = ""
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'bot_start_time' not in st.session_state:
            st.session_state.bot_start_time = None
        self.log_placeholder = st.empty()

    def update_log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.logs = f"[{timestamp}] {message}\n" + st.session_state.logs
        self.log_placeholder.text_area("System Logs", st.session_state.logs, height=300)

    def setup_langchain(self):
        self.prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are LeaveBuddy, a leave management assistant. Be direct and brief.
1. For individual queries: State if person is on leave or present
2. For multiple people: List each person's status
3. Always verify dates
4. Be concise"""),
            HumanMessage(content="{query}")
        ])
        
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt_template,
            verbose=True
        )

    @staticmethod
    @lru_cache(maxsize=1000)
    def get_embedding(text):
        try:
            response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            return tuple(response['data'][0]['embedding'])
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def parse_date(self, date_str):
        try:
            return datetime.strptime(date_str, '%d-%m-%Y')
        except ValueError:
            return None

    def get_leave_status(self, leave_date):
        today = datetime.now()
        leave_dt = self.parse_date(leave_date)
        if not leave_dt:
            return "Invalid Date"
        
        if leave_dt < today:
            return "Completed"
        elif leave_dt > today:
            return "Scheduled"
        else:
            return "Current"

    def create_structured_text(self, row):
        leave_status = self.get_leave_status(row['DATE'])
        return (
            f"{row['NAME']} has leave ({leave_status}) on {row['DATE']} ({row['DAY']}) "
            f"for {row['FESTIVALS']}. This leave is in {row['MONTH']} {row['YEAR']}. "
            f"Status: {leave_status}. Type: {row['FESTIVALS']}. "
            f"Details: {row['NAME']} will be on leave on {row['DAY']}, {row['DATE']} "
            f"for {row['FESTIVALS']} celebration/observation."
        )

    def create_embeddings(self, df):
        records = []
        for i, row in df.iterrows():
            text = self.create_structured_text(row)
            embedding = self.get_embedding(text)
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
                        "year": row['YEAR'],
                        "status": self.get_leave_status(row['DATE'])
                    }
                ))
        return records

    def upload_to_pinecone(self, records):
        try:
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            index = pc.Index(index_name)
            index.upsert(vectors=records)
            return True, "Data successfully uploaded to Pinecone!"
        except Exception as e:
            logger.error(f"Pinecone upload error: {e}")
            return False, f"Upload failed: {str(e)}"

    async def query_pinecone(self, query):
        try:
            index = pc.Index(index_name)
            query_embedding = self.get_embedding(query)
            if query_embedding is None:
                return None
            
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            if results['matches']:
                context = " ".join([match['metadata']['text'] for match in results['matches']])
                return context
            return None
        except Exception as e:
            logger.error(f"Pinecone query error: {e}")
            return None

    async def query_gpt(self, query, context):
        try:
            today = datetime.now().strftime("%d-%m-%Y")
            
            messages = [
                {"role": "system", "content": """You are LeaveBuddy, a leave management assistant. Be direct and brief.
1. For individual queries: State if person is on leave or present
2. For multiple people: List each person's status
3. Always verify dates in context
4. Be concise"""},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=150,
                temperature=0.1,
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            logger.error(f"GPT query error: {e}")
            return f"Error processing query. Please try again."

    async def process_query(self, query):
        try:
            today = datetime.now().strftime("%d-%m-%Y")
            
            if query.lower().startswith("is "):
                name_match = re.search(r"is (\w+)", query, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1)
                    context = await self.query_pinecone(query)
                    if context and name.lower() in context.lower():
                        response = await self.query_gpt(query, context)
                    else:
                        return f"{name} is present today (no leave records found)."
                    return response

            context = await self.query_pinecone(query)
            if context:
                response = await self.query_gpt(query, context)
                return response
            else:
                return "No relevant leave information found."
                
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return "An error occurred. Please try again."

    def setup_slack_handlers(self):
        @app.event("message")
        async def handle_message(event, say):
            try:
                text = event.get("text", "")
                channel = event.get("channel", "")
                
                processing_message = await app.client.chat_postMessage(
                    channel=channel,
                    text="Processing..."
                )
                
                try:
                    response = await asyncio.wait_for(
                        self.process_query(text),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    response = "Request timed out. Please try again."
                
                try:
                    await app.client.chat_update(
                        channel=channel,
                        ts=processing_message['ts'],
                        text=response
                    )
                except SlackApiError as e:
                    logger.error(f"Slack API error: {e}")
                    await say(response)
                
            except Exception as e:
                error_msg = f"Message handling error: {str(e)}"
                logger.error(error_msg)
                await say("Sorry, I encountered an error. Please try again.")

    async def run_slack_bot(self):
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()

    def start_bot_forever(self):
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
        st.title("Leave Buddy - Slack Bot")
        
        st.sidebar.header("📤 Update Leave Data")
        uploaded_file = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                required_columns = ['NAME', 'YEAR', 'MONTH', 'DATE', 'DAY', 'FESTIVALS']
                
                if not all(col in df.columns for col in required_columns):
                    st.sidebar.error(f"Required columns: {', '.join(required_columns)}")
                    return
                
                df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%d-%m-%Y')
                
                st.sidebar.write("📊 Data Preview:")
                st.sidebar.dataframe(df.head())
                
                if st.sidebar.button("🔄 Process and Upload"):
                    with st.spinner("Processing data..."):
                        self.update_log("Creating embeddings...")
                        embeddings = self.create_embeddings(df)
                        
                        self.update_log("Uploading to Pinecone...")
                        success, message = self.upload_to_pinecone(embeddings)
                        
                        if success:
                            st.session_state['data_uploaded'] = True
                            st.session_state.last_update = datetime.now()
                            self.update_log("✅ Upload successful!")
                            st.sidebar.success("✅ Data uploaded successfully!")
                        else:
                            self.update_log(f"❌ Upload failed: {message}")
                            st.sidebar.error(f"❌ Upload failed: {message}")
            
            except Exception as e:
                error_msg = f"Error processing file: {str(e)}"
                logger.error(error_msg)
                self.update_log(f"❌ {error_msg}")
                st.sidebar.error(f"❌ {error_msg}")

        st.header("🤖 Slack Bot Controls")
        
        if st.session_state.last_update:
            st.info(f"📅 Last data update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Bot", disabled=st.session_state.bot_running):
                st.session_state.bot_running = True
                st.session_state.bot_start_time = datetime.now()
                st.write("🔄 Starting Slack bot...")
                self.setup_slack_handlers()
                thread = Thread(target=self.start_bot_forever, daemon=True)
                thread.start()
                st.success("✅ Bot is running! You can now send queries in Slack.")
        
        with col2:
            if st.button("🛑 Stop Bot", disabled=not st.session_state.bot_running):
                st.session_state.bot_running = False
                self.update_log("Bot stopped by user")
                st.warning("⚠️ Bot has been stopped")
                st.rerun()

        if st.session_state.bot_running:
            st.success("🟢 Status: Active and ready for queries")
        else:
            st.warning("🔴 Status: Bot is currently stopped")

def main():
    try:
        st.set_page_config(
            page_title="Leave Buddy - Slack Bot",
            page_icon="🤖",
            layout="wide"
        )
        
        bot = LeaveBot()
        
        st.markdown("""
        ---
        <div style='text-align: center'>
            <p>Leave Buddy Bot - Developed with ❤️ | © 2024</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Application Error: {str(e)}")

if __name__ == "__main__":
    main()
