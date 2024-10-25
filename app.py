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

# Function to query GPT and generate response
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
            model="gpt-4o-mini",  # Updated to GPT-4
            messages=messages,
            max_tokens=500,
            n=1,
            temperature=0.1,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error querying GPT: {e}")
        return f"Error: Unable to process the query. Please try again."

# Function to process query and generate response
async def process_query(query):
    try:
        context = await query_pinecone(query)
        if context:
            response = await query_gpt(query, context)
        else:
            # If no context is found, assume the employee is present
            employee_name = query.split()[1]  # Extracts the name from "is [name] present today?"
            today = datetime.now().strftime("%d-%m-%Y")
            response = f"{employee_name} is present on {today}."
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
        
        # Send initial processing message
        processing_message = await app.client.chat_postMessage(
            channel=channel,
            text="Processing your request... :hourglass_flowing_sand:"
        )
        
        # Get the timestamp of the processing message
        processing_ts = processing_message['ts']
        
        # Process the query
        response = await process_query(text)
        
        # Update the processing message with the final response
        try:
            await app.client.chat_update(
                channel=channel,
                ts=processing_ts,
                text=response
            )
        except SlackApiError as e:
            logger.error(f"Error updating message: {e}")
            # If update fails, send a new message
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
