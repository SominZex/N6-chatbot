import streamlit as st
from groq_model import query_handler

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "charts" not in st.session_state:
    st.session_state.charts = []

st.title("TNS Chatbot(Beta) 🤖")

# Display previous chat history & charts
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if i < len(st.session_state.charts) and st.session_state.charts[i]:
            st.plotly_chart(st.session_state.charts[i], use_container_width=True)

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.charts.append(None) 

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Generating response..."):
        response, chart = query_handler(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.charts.append(chart)

    with st.chat_message("assistant"):
        st.markdown(response)
        if chart:
            st.plotly_chart(chart, use_container_width=True)

if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.charts = []
    st.rerun()
