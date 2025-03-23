import  streamlit as st
import requests

def heath(input_text):
    responce = requests.post('http://localhost:8000/voice/agent/invoke',json={'input':{'tip':input_text}})
    return responce.json()['output']

st.title("TTS to STT")
input_text = st.text_input("Please enter your text")
if input_text:
    st.write(heath(input_text))
