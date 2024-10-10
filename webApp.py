import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# Import your existing function
from main import get_financial_data_analysis

# Import LLM clients
import openai
from ollama import Ollama

# Load environment variables
dotenv_path = Path(".env")
load_dotenv(dotenv_path=dotenv_path)
SIMFIN_API_KEY = os.environ.get("SIMFIN_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Initialize session state for conversation
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'analysis_result' not in st.session_state:
    st.session_state['analysis_result'] = None

# Function to generate system prompt
def get_system_prompt():
    prompt = """
    You are a chat application that will help users analyze financial data.

    You need the 3 following inputs:
    - Stock ticker symbol
    - Year (e.g., 2023)
    - Period (q1, q2, q3, q4, fy)

    Have a conversation with the user to get these inputs. During each iteration, you are going to iteratively build the JSON object with the necessary information.
    
    Once you have everything, thank the user and tell them to give you a moment to analyze the data.

    ##REMEMBER##
    * Convert stock names (e.g. "Apple", "Amazon", or "Google") to their respective ticker symbols (AAPL, AMZN, GOOG).
    * Google and GOOGL must be converted to GOOG
    * Facebook and FB must be converted to META
    * You MUST generate a syntactically correct JSON object

    ##OUTPUT FORMAT##
    Each response MUST be a syntactically correct JSON with the following format:
    {
        "message": string,
        "data": {
            "ticker": string | null
            "year": int | null
            "period": string | null
        }
    }

    You must ask a follow-up question if the user's input is invalid or incomplete. Automatically convert stock names (like Facebook) to their respective ticker symbols (like META).
    """
    return f"{prompt}\nToday's date is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

# Function to initialize the conversation
def initialize_conversation():
    system_message = {
        "role": "system",
        "content": get_system_prompt()
    }
    st.session_state['messages'] = [system_message]

# Function to interact with the LLM
def get_llm_response(user_message, use_ollama=False):
    st.session_state['messages'].append({"role": "user", "content": user_message})

    # Choose the model based on user selection
    if use_ollama:
        client = Ollama()
        model = "llama3.1"
    else:
        openai.api_key = OPENAI_API_KEY
        model = "gpt-4o-mini"

    # Prepare messages for the LLM
    messages = st.session_state['messages']

    if use_ollama:
        # Using Ollama
        response = client.chat(messages=messages, model=model, temperature=0)
        assistant_message = response
    else:
        # Using OpenAI
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0
        )
        assistant_message = response.choices[0].message['content']

    st.session_state['messages'].append({"role": "assistant", "content": assistant_message})
    return assistant_message

# Function to parse JSON from assistant's response
def parse_json_response(response):
    if "{" in response and "}" in response:
        try:
            json_str = response[response.index("{"):response.rindex("}") + 1]
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError:
            return None
    return None

# Streamlit App Layout
st.set_page_config(page_title="Financial Data Analysis AI", page_icon="💹")

st.title("💹 Market Mind")

# Sidebar for selecting LLM
st.sidebar.header("Configuration")
use_ollama = st.sidebar.checkbox("Use Ollama", value=False)
if use_ollama:
    st.sidebar.info("Ensure Ollama is running at the specified URL.")

# Initialize conversation if empty
if not st.session_state['messages']:
    initialize_conversation()

# Display conversation history
st.subheader("Conversation")
for msg in st.session_state['messages']:
    if msg['role'] == 'system':
        st.markdown(f"**System:** {msg['content']}")
    elif msg['role'] == 'user':
        st.markdown(f"**You:** {msg['content']}")
    elif msg['role'] == 'assistant':
        st.markdown(f"**AI:** {msg['content']}")

# Input form for user messages
st.subheader("Your Message")
with st.form(key='input_form', clear_on_submit=True):
    user_input = st.text_input("Type your message here:")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    with st.spinner("AI is responding..."):
        assistant_response = get_llm_response(user_input, use_ollama=use_ollama)
        parsed_data = parse_json_response(assistant_response)

        if parsed_data:
            ticker = parsed_data.get("data", {}).get("ticker")
            year = parsed_data.get("data", {}).get("year")
            period = parsed_data.get("data", {}).get("period")
            message = parsed_data.get("message", "")

            st.session_state['messages'].append({"role": "assistant", "content": message})

            if ticker and year and period:
                st.success("All required information received. Analyzing data...")
                analysis = get_financial_data_analysis(ticker, year, period, use_ollama=use_ollama)
                st.session_state['analysis_result'] = analysis
            else:
                st.info(message)
        else:
            st.error("Failed to parse AI response. Please try again.")
            # Optionally, you can append a system message to remind the AI
            st.session_state['messages'].append({
                "role": "assistant",
                "content": "Oops! I couldn't parse the response. Please provide the information in the correct format."
            })

# Display analysis result if available
if st.session_state['analysis_result']:
    st.subheader("📈 Analysis Result")
    st.write(st.session_state['analysis_result'])

    if st.button("Analyze Another Stock"):
        initialize_conversation()
        st.session_state['analysis_result'] = None
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using [Streamlit](https://streamlit.io/)")

