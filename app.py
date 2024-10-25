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
from aiohttp import ClientError
import json

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

async def format_query_llm(query):
    """First LLM call: Understand and format the query"""
    try:
        messages = [
            {"role": "system", "content": """You are a leave management query analyzer. Your task is to understand and format leave-related queries.

TASK:
Extract and format query information into a standard JSON structure.

INPUT TYPES:
1. Leave Status Questions:
   - "is [name] on leave today?"
   - "will [name] be working tomorrow?"
   - "[name] leave status for [date]?"

2. Leave Reason Questions:
   - "why is [name] taking leave on [date]?"
   - "what's the reason for [name]'s leave?"

3. Period/Duration Questions:
   - "how many days leave does [name] have?"
   - "when is [name]'s next leave?"

OUTPUT FORMAT:
{
    "name": "Employee Name",
    "date": "DD-MM-YYYY",
    "query_type": "status|reason|period",
    "date_type": "today|tomorrow|specific",
    "original_query": "original question"
}

EXAMPLE CONVERSIONS:
1. Input: "is kumar on leave today"
   Output: {
       "name": "Kumar",
       "date": "25-10-2024",
       "query_type": "status",
       "date_type": "today",
       "original_query": "is kumar on leave today"
   }

2. Input: "why is raj taking leave tomorrow"
   Output: {
       "name": "Raj",
       "date": "26-10-2024",
       "query_type": "reason",
       "date_type": "tomorrow",
       "original_query": "why is raj taking leave tomorrow"
   }

RULES:
1. Always capitalize names
2. Convert all dates to DD-MM-YYYY format
3. Use current date for "today"
4. Use next day's date for "tomorrow"
5. Return only the JSON object, no additional text

OUTPUT ONLY THE JSON OBJECT."""},
            {"role": "user", "content": query}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.1
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error in format_query_llm: {e}")
        return None

async def query_pinecone(query_info):
    """Query Pinecone for relevant leave records"""
    try:
        if not query_info:
            return None

        query_json = json.loads(query_info) if isinstance(query_info, str) else query_info
        
        # Create search text
        search_text = f"{query_json['name']} leave {query_json['date']}"
        index = pc.Index(index_name)
        
        query_embedding = get_embedding(search_text)
        if query_embedding is None:
            return None
            
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        if results['matches']:
            return " ".join([match['metadata']['text'] for match in results['matches']])
        return None
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return None

async def analyze_context_llm(context, query_info):
    """Second LLM call: Analyze the context and extract relevant information"""
    try:
        messages = [
            {"role": "system", "content": """You are a leave record analyzer. Your task is to extract relevant leave information from the context.

TASK:
Analyze the context and query details to find relevant leave information.

INPUT:
1. Context: Contains leave records
2. Query Info: JSON with query details

OUTPUT FORMAT:
{
    "found": true/false,
    "leave_info": {
        "name": "Name",
        "date": "DD-MM-YYYY",
        "reason": "Leave reason if found",
        "status": "on_leave|not_on_leave",
        "confidence": "high|medium|low"
    }
}

EXAMPLE:
Context: "John is on leave on 25-12-2024 (Monday) for Christmas."
Query: {"name": "John", "date": "25-12-2024"}
Output: {
    "found": true,
    "leave_info": {
        "name": "John",
        "date": "25-12-2024",
        "reason": "Christmas",
        "status": "on_leave",
        "confidence": "high"
    }
}

RULES:
1. Exact match required for name and date
2. Case-insensitive name matching
3. Only mark as found if date matches exactly
4. Include reason only if explicitly stated
5. Return only the JSON object

OUTPUT ONLY THE JSON OBJECT."""},
            {"role": "user", "content": f"Context: {context}\nQuery Info: {query_info}"}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0.1
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error in analyze_context_llm: {e}")
        return None

async def generate_response_llm(query_info, context_analysis):
    """Third LLM call: Generate natural language response"""
    try:
        messages = [
            {"role": "system", "content": """You are a leave management assistant generating responses. Create clear, natural responses for leave queries.

RESPONSE FORMATS:

1. Leave Status Found (Today):
   → "[Name] is on leave today for [reason]"

2. Leave Status Found (Future):
   → "[Name] will be on leave on [date] for [reason]"

3. Leave Status Found (Past):
   → "[Name] was on leave on [date] for [reason]"

4. No Leave Found (Today):
   → "[Name] is working today"

5. No Leave Found (Future):
   → "[Name] will be working on [date]"

6. No Leave Found (Past):
   → "[Name] was working on [date]"

7. Reason Query (Found):
   → "[Name] is taking leave on [date] for [reason]"

RULES:
1. Use appropriate tense (is/was/will be)
2. Include complete date (DD-MM-YYYY)
3. Always include reason when available
4. Keep responses concise and natural
5. Be definitive - no "might" or "probably"

EXAMPLE RESPONSES:
- "Kumar is on leave today for Diwali celebration"
- "Raj will be working on 26-10-2024"
- "Sarah was on leave on 20-10-2024 for personal reasons"

Generate ONE clear response based on the information provided."""},
            {"role": "user", "content": f"Query Info: {query_info}\nContext Analysis: {context_analysis}"}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error in generate_response_llm: {e}")
        return None

async def validate_response_llm(original_query, generated_response, context_analysis):
    """Fourth LLM call: Validate and polish the response"""
    try:
        messages = [
            {"role": "system", "content": """You are a response validator for leave management queries. Ensure responses are accurate and natural.

VALIDATION CHECKLIST:

1. Format Verification:
   ✓ Contains employee name
   ✓ Uses correct date format
   ✓ Uses appropriate tense
   ✓ Includes reason when available

2. Accuracy Check:
   ✓ Response matches context analysis
   ✓ No contradictions
   ✓ Correct leave status

3. Language Quality:
   ✓ Natural phrasing
   ✓ Clear and concise
   ✓ Professional tone

4. Common Fixes:
   - Add missing dates
   - Correct tense usage
   - Add omitted reasons
   - Fix name capitalization
   - Remove redundant words

EXAMPLE CORRECTIONS:
Input: "kumar will be on leave tomorrow"
Output: "Kumar will be on leave on 26-10-2024 for Diwali celebration"

Return the corrected response or confirm the existing one is correct."""},
            {"role": "user", "content": f"Original Query: {original_query}\nGenerated Response: {generated_response}\nContext Analysis: {context_analysis}"}
        ]
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"Error in validate_response_llm: {e}")
        return generated_response

async def process_query(query):
    try:
        # Stage 1: Format and understand query
        formatted_query = await format_query_llm(query)
        logger.info(f"Formatted query: {formatted_query}")
        
        if not formatted_query:
            return "Please provide a clear question about employee leave."
            
        # Stage 2: Get context
        context = await query_pinecone(formatted_query)
        
        # Stage 3: Analyze context
        context_analysis = await analyze_context_llm(context, formatted_query)
        logger.info(f"Context analysis: {context_analysis}")
        
        # Stage 4: Generate response
        initial_response = await generate_response_llm(formatted_query, context_analysis)
        logger.info(f"Initial response: {initial_response}")
        
        if not initial_response:
            return "I couldn't generate a proper response. Please try again."
            
        # Stage 5: Validate and polish response
        final_response = await validate_response_llm(query, initial_response, context_analysis)
        logger.info(f"Final response: {final_response}")
        
        return final_response if final_response else initial_response
            
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return "I encountered an error while processing your query. Please try again later."

@app.event("message")
async def handle_message(event, say):
    try:
        text = event.get("text", "")
        if not text:
            await say("Please provide a question about employee leave.")
            return

        response = await process_query(text)
        await say(response)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        await say("I'm sorry, I encountered an error. Please try again.")

def run_slack_bot():
    async def start_bot():
        retries = 0
        max_retries = 3
        retry_delay = 5

        while retries < max_retries and st.session_state.bot_running:
            try:
                handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
                await handler.start_async()
                return
            except Exception as e:
                retries += 1
                logger.error(f"Connection attempt {retries} failed: {str(e)}")
                if retries < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max retries reached. Could not start the bot.")
                    st.session_state.bot_running = False
                    st.error("Failed to start Slack bot. Please check your credentials and try again.")
                    return

    try:
        asyncio.run(start_bot())
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")
        st.session_state.bot_running = False
        st.error("An error occurred while running the bot. Please try again.")

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
        try:
            st.session_state.bot_running = True
            st.info("Starting Slack bot...")
            thread = Thread(target=run_slack_bot)
            thread.start()
            time.sleep(2)  # Give some time for connection to establish
            if st.session_state.bot_running:
                st.success("Slack bot is running! You can now ask questions in your Slack channel.")
        except Exception as e:
            st.session_state.bot_running = False
            st.error(f"Failed to start bot: {str(e)}")

# Stop Button
with col2:
    if st.button("Stop Slack Bot", disabled=not st.session_state.bot_running):
        try:
            st.session_state.bot_running = False
            st.info("Stopping Slack bot...")
            # Force stop the bot by rerunning the page
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error stopping bot: {str(e)}")
        finally:
            st.success("Slack bot has been stopped.")

# Display bot status and usage instructions
if st.session_state.bot_running:
    st.markdown("""
    ### Bot Status: 🟢 Active
    
    The bot is running and ready to answer queries in your Slack channel.
    
    #### Example Questions:
    - "Is [name] on leave today?"
    - "Why is [name] taking leave tomorrow?"
    - "Will [name] be working on 25th December?"
    - "[name] leave status for 25-10-2024"
    
    The bot will respond with accurate leave information based on your uploaded data.
    """)
else:
    st.markdown("""
    ### Bot Status: ⭕ Stopped
    
    Click 'Start Slack Bot' to activate the leave management bot.
    
    #### Before Starting:
    1. Upload your leave data Excel file
    2. Process and upload the data
    3. Start the bot
    """)

# Add usage metrics if needed
if st.session_state.bot_running and st.sidebar.checkbox("Show Usage Metrics"):
    st.sidebar.markdown("""
    ### Usage Metrics
    - Queries Processed: 0
    - Last Query Time: N/A
    - Average Response Time: N/A
    """)

# Footer
st.markdown("---")
st.markdown("Leave Buddy Bot © 2024 | Powered by GPT-4 and Pinecone")

if __name__ == "__main__":
    st.write("Leave Buddy is ready to use!")
