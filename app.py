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
from dotenv import load_dotenv
import os
import re

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app title
st.title("Leave Buddy - Slack Bot")

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "annual-leave"

# Initialize Slack app
app = AsyncApp(token=SLACK_BOT_TOKEN)

# Create a placeholder for logs
if 'logs' not in st.session_state:
    st.session_state.logs = ""
log_placeholder = st.empty()

def update_log(message):
    st.session_state.logs += message + "\n"
    log_placeholder.text_area("Logs", st.session_state.logs, height=300)

@lru_cache(maxsize=1000)
def get_embedding(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return tuple(response['data'][0]['embedding'])
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def create_embeddings(df):
    records = []
    for i, row in df.iterrows():
        text = f"{row['NAME']} is on leave on {row['DATE']} ({row['DAY']}) for {row['FESTIVALS']}. This is in {row['MONTH']} {row['YEAR']}."
        embedding = get_embedding(text)
        if embedding:
            records.append((str(i), embedding, {"text": text}))
    return records

def upload_to_pinecone(records):
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

async def query_pinecone(query):
    try:
        index = pc.Index(index_name)
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return None
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        if results['matches']:
            context = " ".join([match['metadata']['text'] for match in results['matches']])
            return context
        else:
            return None
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
    return None

async def analyze_query(query):
    """Pre-process and analyze the query to determine intent and extract key information"""
    try:
        messages = [
            {"role": "system", "content": """You are a leave management query analyzer. Extract key information from queries.

TASK: Analyze the query and output in JSON format with these fields:
{
    "intent": "status|reason|period",
    "name": "employee name",
    "date": "DD-MM-YYYY",
    "formatted_query": "standardized query"
}

EXAMPLES:

Query: "is kumar on leave today"
{
    "intent": "status",
    "name": "Kumar",
    "date": "25-10-2024",
    "formatted_query": "check Kumar leave status for 25-10-2024"
}

Query: "why is raj taking leave tomorrow"
{
    "intent": "reason",
    "name": "Raj",
    "date": "26-10-2024",
    "formatted_query": "check Raj leave reason for 26-10-2024"
}

RULES:
1. Convert all dates to DD-MM-YYYY format
2. Capitalize names
3. Never add information not in query
4. Format dates: today → current date, tomorrow → next date"""},
            {"role": "user", "content": query}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        return None

async def generate_response(query_info, context):
    """Generate response based on analyzed query and context"""
    try:
        messages = [
            {"role": "system", "content": """You are a leave management assistant specialized in providing clear, accurate responses.

CONTEXT RULES:
1. Only use information explicitly stated in the context
2. Never assume or add information
3. If leave info exists in context → person is on leave
4. If no leave info in context → person is not on leave

RESPONSE FORMATS:

1. For Status Queries:
   Found in context:
   → "[Name] will be on leave on [date] for [reason]"
   Not found:
   → "[Name] will be working on [date]"

2. For Reason Queries:
   Found in context:
   → "[Name] has planned leave on [date] for [reason]"
   Not found:
   → "[Name] will be working on [date]"

3. For Period Queries:
   → List all leave dates found in context

EXAMPLES:

Context: "Kumar is on leave on 25-10-2024 (Friday) for Diwali"
Query: Status check
Response: "Kumar will be on leave on 25-10-2024 for Diwali"

Context: No matching info
Query: Status check
Response: "Kumar will be working on 25-10-2024"

KEY POINTS:
- Use "will be on leave" for future dates
- Use "is on leave" for current date
- Use "will be working" when no leave found
- Always include full date (DD-MM-YYYY)
- Include exact reason if available
- Keep responses concise and clear"""},
            {"role": "user", "content": f"Query Info: {query_info}\nContext: {context}"}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.1
        )
        
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None

async def validate_response(response, query_info, context):
    """Validate and ensure response quality"""
    try:
        messages = [
            {"role": "system", "content": """You are a response validator for leave management system. Ensure responses meet all criteria.

VALIDATION RULES:

1. Response Structure:
   ✓ Contains employee name exactly as in query
   ✓ Has date in DD-MM-YYYY format
   ✓ Uses correct phrasing:
     - "will be on leave" for future dates
     - "is on leave" for current date
     - "will be working" when no leave found
   ✓ Includes reason when available

2. Content Accuracy:
   ✓ Matches context information exactly
   ✓ No conflicting information
   ✓ No missing required details

3. Format Check:
   For Leave Found:
   → "[Name] will be on leave on [DD-MM-YYYY] for [reason]"
   For No Leave:
   → "[Name] will be working on [DD-MM-YYYY]"

4. Fix Common Issues:
   - Missing date → Add full date
   - Wrong format → Standardize format
   - Inconsistent status → Match context
   - Missing reason → Add if available

Return corrected response if needed."""},
            {"role": "user", "content": f"Response: {response}\nQuery Info: {query_info}\nContext: {context}"}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.1
        )
        
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error validating response: {e}")
        return response

async def process_query(query):
    try:
        # Stage 1: Analyze query
        query_info = await analyze_query(query)
        logger.info(f"Query analysis: {query_info}")
        
        # Stage 2: Get context
        context = await query_pinecone(query)
        
        # Stage 3: Generate initial response
        initial_response = await generate_response(query_info, context)
        logger.info(f"Initial response: {initial_response}")
        
        # Stage 4: Validate and format final response
        final_response = await validate_response(initial_response, query_info, context)
        logger.info(f"Final response: {final_response}")
        
        return final_response or "I apologize, but I couldn't process your query. Please try rephrasing it."
            
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return "I encountered an error while processing your query. Please try again later."

@app.event("message")
async def handle_message(event, say):
    try:
        text = event.get("text", "")
        response = await process_query(text)
        await say(response)
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        await say("I'm sorry, I encountered an error. Please try again.")

def run_slack_bot():
    async def start_bot():
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()
    asyncio.run(start_bot())

# Streamlit UI
st.sidebar.header("Update Leave Data")
uploaded_file = st.sidebar.file_uploader("Upload Excel file", type="xlsx")
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%d-%m-%Y')
    
    st.sidebar.write("Uploaded Data Preview:")
    st.sidebar.dataframe(df.head())
    
    if st.sidebar.button("Process and Upload Data"):
        with st.spinner("Processing and uploading data..."):
            embeddings = create_embeddings(df)
            success, message = upload_to_pinecone(embeddings)
        st.sidebar.write(message)
        if success:
            st.session_state['data_uploaded'] = True
            st.sidebar.success("Data processed and uploaded successfully!")

# Main interface for Slack bot controls
st.header("Slack Bot Controls")

# Initialize bot_running state if not exists
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

# Create two columns for Start and Stop buttons
col1, col2 = st.columns(2)

# Start Button
with col1:
    if st.button("Start Slack Bot", disabled=st.session_state.bot_running):
        st.session_state.bot_running = True
        st.write("Starting Slack bot...")
        thread = Thread(target=run_slack_bot)
        thread.start()
        st.success("Slack bot is running! You can now ask questions in your Slack channel.")

# Stop Button
with col2:
    if st.button("Stop Slack Bot", disabled=not st.session_state.bot_running):
        st.session_state.bot_running = False
        st.write("Stopping Slack bot...")
        # Force stop the bot by rerunning the page
        st.experimental_rerun()
        st.success("Slack bot has been stopped.")

# Display bot status
if st.session_state.bot_running:
    st.write("Slack bot is active and ready to answer queries.")
else:
    st.write("Slack bot is stopped.")

if __name__ == "__main__":
    st.write("Leave Buddy is ready to use!")
