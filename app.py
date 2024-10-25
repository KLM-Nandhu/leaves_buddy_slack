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

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "annual-leave"

# Initialize Slack
app = AsyncApp(token=SLACK_BOT_TOKEN)

class LeaveBot:
    def __init__(self):
        self.initialize_state()
        self.setup_ui()
        
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
            f"for {row['FESTIVALS']} celebration/observation. "
            f"This is a {row['FESTIVALS']} holiday in {row['MONTH']} {row['YEAR']}. "
            f"Leave Status: {leave_status}. Employee Name: {row['NAME']}. "
            f"Date Information: {row['DATE']} ({row['DAY']})."
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
                top_k=15,
                include_metadata=True
            )
            
            if results['matches']:
                matches = sorted(
                    results['matches'],
                    key=lambda x: datetime.strptime(x['metadata']['date'], '%d-%m-%Y')
                )
                
                completed = []
                scheduled = []
                current = []
                
                for match in matches:
                    status = match['metadata'].get('status', '')
                    if status == 'Completed':
                        completed.append(match['metadata']['text'])
                    elif status == 'Scheduled':
                        scheduled.append(match['metadata']['text'])
                    elif status == 'Current':
                        current.append(match['metadata']['text'])
                
                context_parts = []
                if completed:
                    context_parts.append("COMPLETED LEAVES: " + " ".join(completed))
                if current:
                    context_parts.append("CURRENT LEAVES: " + " ".join(current))
                if scheduled:
                    context_parts.append("SCHEDULED LEAVES: " + " ".join(scheduled))
                
                return " | ".join(context_parts)
            return None
        except Exception as e:
            logger.error(f"Pinecone query error: {e}")
            return None

    async def query_gpt(self, query, context):
        try:
            today = datetime.now().strftime("%d-%m-%Y")
            
            messages = [
                {"role": "system", "content": f"""You are LeaveBuddy, a precise leave management assistant. Today is {today}. Follow these rules exactly:

1. QUERY ANALYSIS:
   - Carefully analyze if query is about past, present, or future dates
   - Identify single or multiple employee requests
   - Determine if date-specific or general inquiry
   - Check if asking for comparison or individual status

2. RESPONSE FORMAT:
   Always structure as follows:
   
   🤖 Processing request...
   
   📅 Time Period: [Past/Present/Future]
   
   RESULTS:
   [Employee Name]
   ✓ Completed Leaves:
   - DD-MM-YYYY ([Day]): [Festival]
   
   📍 Current Status:
   - [Present/On Leave] Today
   
   🗓 Scheduled Leaves:
   - DD-MM-YYYY ([Day]): [Festival]
   
   📊 Summary:
   - Total Past Leaves: X
   - Total Scheduled Leaves: Y
   - Total Days: Z

3. COMPARISON FORMAT:
   When comparing employees:
   
   COMPARATIVE ANALYSIS:
   📋 Common Leaves:
   - [Date]: [Festival] (Employees: [Names])
   
   🔄 Individual Patterns:
   [Name1]:
   - Unique leaves: [Dates]
   [Name2]:
   - Unique leaves: [Dates]
   
   📈 Statistics:
   - Most leaves: [Name]
   - Upcoming leaves: [Count] per person

4. RULES:
   - Always show exact dates (DD-MM-YYYY)
   - Include day of week when available
   - Use emojis for better readability
   - Sort leaves chronologically
   - Separate past and future leaves
   - Show total leave counts
   - Verify all data matches context
   - Complete all responses fully
   - Never cut off mid-response

5. SPECIAL HANDLING:
   - No data: "No leave records found for [Name]"
   - Partial data: "Showing available records for [Date Range]"
   - Invalid queries: "Please specify [missing info]"
   - Multiple names: List each person separately
   - Date conflicts: Show all overlapping leaves

6. VERIFICATION STEPS:
   - Cross-check all dates
   - Verify festival names
   - Confirm employee names
   - Double-check leave counts
   - Ensure response completeness

Remember: Every response must be complete, accurate, and well-formatted with all relevant information."""},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
                n=1,
                temperature=0.1,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            logger.error(f"GPT query error: {e}")
            return f"Error processing query. Please try again."

    async def process_query(self, query):
        try:
            logger.info(f"Processing query: {query}")
            context = await self.query_pinecone(query)
            
            if context:
                logger.info(f"Found context: {context[:200]}...")
                response = await self.query_gpt(query, context)
                logger.info(f"Generated response: {response}")
                return response
            else:
                name_match = re.search(r"is (\w+)", query, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1)
                    today = datetime.now().strftime("%d-%m-%Y")
                    return (
                        f"🤖 Processing request...\n\n"
                        f"📅 Date: {today}\n\n"
                        f"RESULTS:\n"
                        f"{name} is present today (no leave records found).\n\n"
                        f"📊 Summary:\n"
                        f"- No leave records found for specified date"
                    )
                else:
                    return "Please provide a specific employee name or date for leave information."
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
                    text="🤖 Processing your request... :hourglass_flowing_sand:"
                )
                
                self.update_log(f"New query: {text}")
                response = await self.process_query(text)
                
                try:
                    await app.client.chat_update(
                        channel=channel,
                        ts=processing_message['ts'],
                        text=response
                    )
                    self.update_log(f"Response sent: {response[:100]}...")
                except SlackApiError as e:
                    logger.error(f"Slack API error: {e}")
                    await say(response)
                
            except Exception as e:
                error_msg = f"Message handling error: {str(e)}"
                logger.error(error_msg)
                self.update_log(error_msg)
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
        
        # Sidebar
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

        # Main interface
        st.header("🤖 Slack Bot Controls")
        
        # Display last update time
        if st.session_state.last_update:
            st.info(f"📅 Last data update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Bot control buttons
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
                st.experimental_rerun()

        # Bot status indicator
        if st.session_state.bot_running:
            st.success("🟢 Status: Active and ready for queries")
            
            # Display usage instructions
            with st.expander("📖 Usage Instructions"):
                st.markdown("""
                ### How to Use Leave Buddy
                
                1. **Basic Queries**:
                   - `is [name] on leave today?`
                   - `when is [name]'s next leave?`
                   - `show all leaves for [name]`
                
                2. **Multiple Employee Queries**:
                   - `compare leaves for [name1] and [name2]`
                   - `who is on leave today?`
                   - `show all upcoming leaves`
                
                3. **Date-Specific Queries**:
                   - `who is on leave on [date]?`
                   - `show leaves in [month]`
                   - `is anyone on leave next week?`
                
                4. **Statistical Queries**:
                   - `how many leaves does [name] have?`
                   - `show leave summary for team`
                   - `who has the most leaves?`
                """)
        else:
            st.warning("🔴 Status: Bot is currently stopped")

        # System monitoring
        with st.expander("🔍 System Monitoring"):
            st.markdown("### System Status")
            
            # Display current time
            st.write(f"🕒 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Bot uptime if running
            if st.session_state.bot_running and st.session_state.get('bot_start_time'):
                uptime = datetime.now() - st.session_state.bot_start_time
                st.write(f"⏱️ Bot Uptime: {str(uptime).split('.')[0]}")
            
            # Display API status
            try:
                if openai.api_key and PINECONE_API_KEY and SLACK_BOT_TOKEN:
                    st.success("✅ All API keys configured")
                else:
                    st.error("❌ Missing API keys")
            except Exception:
                st.error("❌ Error checking API configuration")

def main():
    try:
        st.set_page_config(
            page_title="Leave Buddy - Slack Bot",
            page_icon="🤖",
            layout="wide"
        )
        
        # Initialize the bot
        bot = LeaveBot()
        
        # Add footer
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
