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
from langchain.callbacks.tracers import LangChainTracer, ConsoleCallbackHandler
from langchain.smith import RunEvalConfig
from langchain.callbacks import tracing_enabled
from langchain.callbacks.manager import CallbackManager
from langsmith import Client
import langsmith

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

# Initialize LangSmith client
langsmith_client = Client()

# Initialize tracers
console_handler = ConsoleCallbackHandler()
tracer = LangChainTracer(
    project_name="Leaves-Buddy",
    client=langsmith_client
)

# Initialize ChatOpenAI with tracing
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.1,
    streaming=False,
    request_timeout=10,
    callbacks=[console_handler, tracer]
)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "annual-leave"

# Initialize Slack
app = AsyncApp(token=SLACK_BOT_TOKEN)

SYSTEM_PROMPT = """You are LeaveBuddy, a precise leave management assistant. Today is {current_date}. Follow these rules exactly:

1. DATA ACCURACY:
   - Only use leave information explicitly present in the context
   - Never make assumptions about leaves not in the data
   - Always verify dates, names, and reasons before responding
   - If a person is not shown as on leave for a date, they are present

2. QUERY HANDLING:
   For individual queries ("is [name] on leave?"):
   - Check exact dates in context
   - Response format: "[Name] is on leave on [Date] for [Festival]" or "[Name] is present on [Date]"

3. MULTIPLE EMPLOYEE QUERIES:
   When asked about multiple people:
   - List each person separately
   - Show all leave dates for each person
   - Compare any overlapping leaves
   - Format:
     [Name1]:
     - [Date]: [Festival]
     - [Date]: [Festival]
     [Name2]:
     - [Date]: [Festival]

4. DATE VERIFICATION:
   - Use exact DD-MM-YYYY format
   - Include day of week if available
   - For future dates, explicitly check if leave is scheduled

5. ERROR HANDLING:
   If information is missing:
   - Name not found: "No records found for [Name]"
   - Date not found: "No leave information for that date"
   - Unclear query: "Could you please specify the name and date?"

6. COMPARATIVE ANALYSIS:
   When comparing leaves:
   - List all relevant dates
   - Note overlapping leaves
   - Highlight any differences

7. RESPONSE STRUCTURE:
   - Start with "Processing request..."
   - Provide clear, direct answers
   - Use bullet points for multiple items
   - End with verification statement

8. VERIFICATION:
   - Double-check all dates and names
   - Verify festival/reason matches
   - Ensure response matches context exactly
   
9. FORMAT RULES:
   - Always show dates as DD-MM-YYYY
   - Include weekday when available
   - Show total leave count at the end
   - List leaves chronologically
   - Use clear sections for past and future leaves
   - Include emojis for better readability:
     📅 for dates
     ✓ for completed leaves
     🗓 for upcoming leaves
     ❌ for no leaves
     👥 for multiple people

Remember:
1. Be precise and accurate
2. Only use information from the provided context
3. Always verify before responding
4. Keep responses complete and well-structured"""

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
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="{query}")
        ])
        
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt_template,
            verbose=True,
            callbacks=[console_handler, tracer]
        )

    def check_langsmith_connection(self):
        try:
            projects = langsmith_client.list_projects()
            current_project = next((p for p in projects if p.name == "Leaves-Buddy"), None)
            
            if current_project:
                self.update_log("✅ LangSmith connected successfully")
                return True
            else:
                self.update_log("❌ LangSmith project not found")
                return False
        except Exception as e:
            self.update_log(f"❌ LangSmith connection error: {str(e)}")
            return False

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
            with tracing_enabled() as session:
                today = datetime.now().strftime("%d-%m-%Y")
                
                session.metadata = {
                    "query_type": "leave_query",
                    "timestamp": today,
                    "has_context": bool(context)
                }
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT.format(current_date=today)},
                    {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
                ]
                
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.1,
                    presence_penalty=0.0,
                    frequency_penalty=0.0
                )
                
                session.log({
                    "response": response.choices[0].message['content'],
                    "model": "gpt-4o-mini",
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None
                })
                
                return response.choices[0].message['content'].strip()
        except Exception as e:
            logger.error(f"GPT query error: {e}")
            return f"Error processing query. Please try again."

    async def process_query(self, query):
        try:
            with tracing_enabled() as session:
                session.metadata = {
                    "query_text": query,
                    "timestamp": datetime.now().isoformat()
                }
                
                today = datetime.now().strftime("%d-%m-%Y")
                
                if query.lower().startswith("is "):
                    name_match = re.search(r"is (\w+)", query, re.IGNORECASE)
                    if name_match:
                        name = name_match.group(1)
                        context = await self.query_pinecone(query)
                        session.log({"found_context": bool(context)})
                        
                        if context and name.lower() in context.lower():
                            response = await self.query_gpt(query, context)
                        else:
                            response = f"{name} is present today (no leave records found)."
                        return response

                context = await self.query_pinecone(query)
                session.log({"found_context": bool(context)})
                
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
                with tracing_enabled() as session:
                    text = event.get("text", "")
                    channel = event.get("channel", "")
                    
                    session.metadata = {
                        "channel": channel,
                        "message_type": "slack_query",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    processing_message = await app.client.chat_postMessage(
                        channel=channel,
                        text="🤖 Processing your request... :hourglass_flowing_sand:"
                    )
                    
                    try:
                        response = await asyncio.wait_for(
                            self.process_query(text),
                            timeout=10.0
                        )
                        
                        session.log({
                            "status": "success",
                            "response_time": time.time(),
                            "response_length": len(response)
                        })
                        
                    except asyncio.TimeoutError:
                        response = "Request timed out. Please try again."
                        session.log({"status": "timeout"})
                    
                    try:
                        await app.client.chat_update(
                            channel=channel,
                            ts=processing_message['ts'],
                            text=response
                        )
                    except SlackApiError as e:
                        logger.error(f"Slack API error: {e}")
                        await say(response)
                        session.log({"status": "slack_api_error"})
                    
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

        # System status and monitoring
        with st.expander("🔍 System Monitoring"):
            st.markdown("### System Status")
            
            # Display current time
            st.write(f"🕒 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Bot uptime if running
            if st.session_state.bot_running and st.session_state.get('bot_start_time'):
                uptime = datetime.now() - st.session_state.bot_start_time
                st.write(f"⏱️ Bot Uptime: {str(uptime).split('.')[0]}")
            
            # Check API configurations
            st.markdown("### API Status")
            apis_ok = True
            
            # Check OpenAI
            if openai.api_key:
                st.success("✅ OpenAI API Connected")
            else:
                st.error("❌ OpenAI API Key Missing")
                apis_ok = False
            
            # Check Pinecone
            if PINECONE_API_KEY:
                st.success("✅ Pinecone API Connected")
            else:
                st.error("❌ Pinecone API Key Missing")
                apis_ok = False
            
            # Check Slack
            if SLACK_BOT_TOKEN and SLACK_APP_TOKEN:
                st.success("✅ Slack API Connected")
            else:
                st.error("❌ Slack API Keys Missing")
                apis_ok = False
            
            # Check LangSmith
            if self.check_langsmith_connection():
                st.success("✅ LangSmith Connected")
            else:
                st.error("❌ LangSmith Connection Failed")
                apis_ok = False
            
            if apis_ok:
                st.success("✅ All Systems Operational")
            else:
                st.warning("⚠️ Some Systems Need Attention")

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
