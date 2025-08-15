# ui.py

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from backend import get_response, get_chat_history

# --- Page Setup ---
st.set_page_config(page_title="PDF Chatbot", page_icon="💬")
st.title("💬 PDF Chatbot")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = get_chat_history()

# --- Display Chat History ---
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("ai"):
            st.markdown(message.content)

# --- Chat Input ---
user_query = st.chat_input("Ask something about the PDF...")

if user_query:
    # Add user message to UI and LangChain memory
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("human"):
        st.markdown(user_query)

    # Get AI response
    ai_response = get_response(user_query)

    # Add AI response to UI and LangChain memory
    st.session_state.chat_history.append(AIMessage(content=ai_response))
    with st.chat_message("ai"):
        st.markdown(ai_response)
