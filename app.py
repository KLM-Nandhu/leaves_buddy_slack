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

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Set up logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from Streamlit secrets or .env file
def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.getenv(key)

# Initialize API keys
openai.api_key = get_secret("OPENAI_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
SLACK_BOT_TOKEN = get_secret("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = get_secret("SLACK_APP_TOKEN")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "annual-leave"

# Initialize Slack app
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
        self.log_placeholder = st.empty()

    def update_log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.logs = f"[{timestamp}] {message}\n" + st.session_state.logs
        self.log_placeholder.text_area("Logs", st.session_state.logs, height=300)

    @staticmethod
    @lru_cache(maxsize=1000)
    def get_embedding(text):
        try:
            response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
            return tuple(response['data'][0]['embedding'])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def create_structured_text(self, row):
        """Create a well-structured text representation of leave data"""
        return (
            f"{row['NAME']} has leave scheduled for {row['DATE']} ({row['DAY']}) "
            f"for {row['FESTIVALS']}. This leave is in {row['MONTH']} {row['YEAR']}. "
            f"The day is a {row['DAY']}. Festival/Event: {row['FESTIVALS']}. "
            f"Employee: {row['NAME']}. Full date: {row['DATE']}. "
            f"This is a {row['FESTIVALS']} holiday."
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
                        "year": row['YEAR']
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
        try:
            index = pc.Index(index_name)
            query_embedding = self.get_embedding(query)
            if query_embedding is None:
                return None
                
            # Increase top_k for better context
            results = index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            if results['matches']:
                # Sort matches by date for chronological order
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

    def parse_date(self, date_str):
        """Parse date string to consistent format"""
        try:
            return datetime.strptime(date_str, '%d-%m-%Y')
        except ValueError:
            return None

    async def query_gpt(self, query, context):
        try:
            today = datetime.now().strftime("%d-%m-%Y")
            
            messages = [
                {"role": "system", "content": f"""You are LeaveBuddy, a precise leave management assistant. Today is {today}. Follow these rules exactly:

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
   - Ensure response matches context exactly"""},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                n=1,
                temperature=0.1  # Low temperature for consistent responses
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            logger.error(f"Error querying GPT: {e}")
            return f"Error: Unable to process the query. Please try again."

    async def process_query(self, query):
        try:
            # Log the incoming query
            logger.info(f"Processing query: {query}")
            
            # Get context from Pinecone
            context = await self.query_pinecone(query)
            
            if context:
                # Log the context found
                logger.info(f"Context found: {context[:200]}...")
                
                # Get response from GPT
                response = await self.query_gpt(query, context)
                
                # Log the response
                logger.info(f"Generated response: {response}")
                
                return response
            else:
                # Handle case when no context is found
                # Extract name from query if possible
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
                
                # Log the request
                self.update_log(f"Received query: {text}")
                
                # Process the query
                response = await self.process_query(text)
                
                # Update the message with the response
                try:
                    await app.client.chat_update(
                        channel=channel,
                        ts=processing_message['ts'],
                        text=response
                    )
                    
                    # Log the successful response
                    self.update_log(f"Responded: {response}")
                    
                except SlackApiError as e:
                    logger.error(f"Error updating message: {e}")
                    await say(response)
                    
            except Exception as e:
                error_msg = f"Error in handle_message: {str(e)}"
                logger.error(error_msg)
                self.update_log(error_msg)
                await say("I'm sorry, I encountered an error. Please try again.")

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
        
        # Sidebar for data upload
        st.sidebar.header("Update Leave Data")
        uploaded_file = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                required_columns = ['NAME', 'YEAR', 'MONTH', 'DATE', 'DAY', 'FESTIVALS']
                
                # Validate columns
                if not all(col in df.columns for col in required_columns):
                    st.sidebar.error("Excel file must contain all required columns: " + ", ".join(required_columns))
                    return
                
                # Convert date to consistent format
                df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%d-%m-%Y')
                
                st.sidebar.write("Uploaded Data Preview:")
                st.sidebar.dataframe(df.head())
                
                if st.sidebar.button("Process and Upload Data"):
                    with st.spinner("Processing and uploading data..."):
                        # Create embeddings
                        self.update_log("Creating embeddings...")
                        embeddings = self.create_embeddings(df)
                        
                        # Upload to Pinecone
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

        # Main interface
        st.header("Slack Bot Controls")
        
        # Show last update time if available
        if st.session_state.last_update:
            st.info(f"Last data update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if st.button("Start Slack Bot", disabled=st.session_state.bot_running):
            st.session_state.bot_running = True
            st.write("Starting Slack bot...")
            self.setup_slack_handlers()
            thread = Thread(target=self.start_bot_forever, daemon=True)
            thread.start()
            st.success("Slack bot is running! You can now ask questions in your Slack channel.")

        if st.session_state.bot_running:
            st.write("Status: Active and ready for queries")
            if st.button("Stop Bot"):
                st.session_state.bot_running = False
                st.experimental_rerun()

def main():
    bot = LeaveBot()

if __name__ == "__main__":
    main()
