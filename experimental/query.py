from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from dotenv import load_dotenv
from langsmith import trace

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Define model path 
model_path = "./model"

# Use Streamlit's caching to load the tokenizer and model
@st.cache_resource
def load_tokenizer_and_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_tokenizer_and_model(model_path)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, please respond to the user query."),
        ("user", "Question: {question}")
    ]
)

# Streamlit UI
st.title("Demo TNS AI")
input_text = st.text_input("Search the topic you want")

output_parser = StrOutputParser()

def generate_response(question):
    # Format the prompt
    formatted_prompt = prompt.format(question=question)
    print(f"Formatted Prompt: {formatted_prompt}")  
    
    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    # Ensure inputs are tensors
    if not isinstance(inputs['input_ids'], torch.Tensor):
        raise TypeError("Tokenized inputs must be a tensor.")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    
    # Decode the response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

if input_text:
    try:
        response = generate_response(input_text)
        st.write(response)
        # Log the response for tracing
        print(f"Generated response: {response}")
        # # Log the prompt and response to LangSmith
        # tracer.log_prompt(formatted_prompt)
        # tracer.log_response(response)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        print(f"Error during response generation: {e}")
    # finally:
    #     # End the trace
    #     tracer.end_trace()