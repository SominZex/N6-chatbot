import requests
import streamlit as st

def get_llama_response(input_text):
    try:
        response = requests.post("http://localhost:8000/essay/invoke",
                                 json={"input": {'topic': input_text}})
        response.raise_for_status()
        data = response.json()
        
        # Adjusted to access the 'output' key directly
        if 'output' in data:
            return data['output']
        else:
            return "Unexpected response format: " + str(data)
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    except ValueError:
        return "Invalid JSON response"

st.title("Langchain API Call")
input_text = st.text_input("Write an essay on")
input_text2 = st.text_input("Write a poem on")

if input_text:
    st.write(get_llama_response(input_text))

if input_text2:
    st.write(get_llama_response(input_text2))