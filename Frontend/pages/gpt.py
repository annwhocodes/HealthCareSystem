import streamlit as st
from PIL import Image
import google.generativeai as genai
from navbar import navbar
from footer import footer
from crew_orchestrator import process_medical_query
from styles import load_styles

# Set Page Configuration
st.set_page_config(
    page_title="MediMind AI | Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom styles
load_styles()

# Set Gemini API Key
GOOGLE_API_KEY = "AIzaSyBMQzWLHcTOTNzDHQoCB-rD0B2x558RyrA"
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display navbar
navbar()

# Main chat interface
st.title("🏥 MediMind AI Healthcare Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you with your medical questions?"):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            try:
                # Use the crew orchestration to process the query
                response = process_medical_query(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Display footer
footer()