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

# Function to update Streamlit log
def update_log(message):
    st.session_state.logs += message + "\n"
    log_placeholder.text_area("Logs", st.session_state.logs, height=300)

# Cached function to generate embeddings
@lru_cache(maxsize=1000)
def get_embedding(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return tuple(response['data'][0]['embedding'])
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

# Function to create embeddings
def create_embeddings(df):
    records = []
    for i, row in df.iterrows():
        text = f"{row['NAME']} is on leave on {row['DATE']} ({row['DAY']}) for {row['FESTIVALS']}. This is in {row['MONTH']} {row['YEAR']}."
        embedding = get_embedding(text)
        if embedding:
            records.append((str(i), embedding, {"text": text}))
    return records

# Function to upload data to Pinecone
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

# Function to query Pinecone and format response
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

# Function to extract date from query
def extract_date_from_query(query):
    # First, look for DD-MM-YYYY format
    import re
    date_pattern = r'\d{2}-\d{2}-\d{4}'
    matches = re.findall(date_pattern, query)
    if matches:
        return matches[0]
    
    # Look for specific date mentions
    words = query.split()
    for word in words:
        if word.replace('th','').replace('st','').replace('nd','').replace('rd','').isdigit():
            # If found a day number, convert it to proper format
            try:
                day = int(word.replace('th','').replace('st','').replace('nd','').replace('rd',''))
                current_date = datetime.now()
                return current_date.replace(day=day).strftime('%d-%m-%Y')
            except ValueError:
                continue
    
    # If no date found, return today's date
    return datetime.now().strftime('%d-%m-%Y')

# Function to extract name from query
def extract_name_from_query(query):
    # Split query into words and look for name
    words = query.lower().split()
    # Skip common words
    skip_words = {'is', 'has', 'leave', 'on', 'why', 'taking', 'today', 'tomorrow', 'yesterday'}
    for word in words:
        if word not in skip_words and not any(char.isdigit() for char in word):
            return word.capitalize()
    return None

# Improved query_gpt function
async def query_gpt(query, context):
    try:
        today = datetime.now().strftime("%d-%m-%Y")
        
        messages = [
            {"role": "system", "content": f"""You are LeaveBuddy, an efficient AI assistant for employee leave information. Today is {today}. Follow these rules strictly:

1. Provide concise, direct answers about employee leaves.
2. provide processing message for every request of the user.
3. Always mention specific dates in your responses.
4. For queries about total leave days, use this format:
   [Employee Name] has [X] total leave days in [Year]:
   - [Date]: [Reason]
   - [Date]: [Reason]
   ...
   Total: [X] days
5. For presence queries:
   - If leave information is found for the date, respond with: if the information is found that person will not appear on that day
     "[Employee Name] is present on [Date]. Reason: [Leave Reason]"
   - If no leave information is found for the date, respond with:if the information is not found that person should appear on that day
     "[Employee Name] is   not present on [Date]."
6. IMPORTANT: Absence of leave information in the database means the employee is present.
7. Only mention leave information if it's explicitly stated in the context.
8. Limit responses to essential information only.
9. Do not add any explanations or pleasantries.
10. in final answer check again in DB is it correct ?
11. if the question is overall like example : is anyboday leave today ?
    - check the date in DB and give the solution"""},
            {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            n=1,
            temperature=0.3,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error querying GPT: {e}")
        return f"Error: Unable to process the query. Please try again."

# Improved process_query function
async def process_query(query):
    try:
        context = await query_pinecone(query)
        name = extract_name_from_query(query)
        date = extract_date_from_query(query)
        
        if not name:
            return "Please specify the employee name in your query."
            
        if context:
            response = await query_gpt(query, context)
        else:
            response = f"{name} is not on leave on {date}."
            
        # Clean up response
        response = response.replace("  ", " ").strip()
        if response.lower().startswith("response:"):
            response = response[9:].strip()
            
        return response
        
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return "I encountered an error while processing your query. Please try again later."

# Enhanced Slack event handler with fixed loading message
@app.event("message")
async def handle_message(event, say):
    try:
        text = event.get("text", "")
        channel = event.get("channel", "")
        
        # Process the query
        response = await process_query(text)
        
        # Send the response
        await say(response)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        await say("I'm sorry, I encountered an error. Please try again.")

# Function to run the Slack bot
def run_slack_bot():
    async def start_bot():
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()

    asyncio.run(start_bot())

# Sidebar for optional data upload
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

# Main interface for starting the Slack bot
st.header("Slack Bot Controls")
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

if st.button("Start Slack Bot", disabled=st.session_state.bot_running):
    st.session_state.bot_running = True
    st.write("Starting Slack bot...")
    thread = Thread(target=run_slack_bot)
    thread.start()
    st.success("Slack bot is running! You can now ask questions in your Slack channel.")

if st.session_state.bot_running:
    st.write("Slack bot is active and ready to answer queries.")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Leave Buddy is ready to use!")
