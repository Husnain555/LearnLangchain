import requests
import streamlit as st

def get_llama(input_text):
    response = requests.post("http://localhost:8000/essay/invoke", json={'input': {'topic': input_text}})
    return response.json()['output']
st.title('client side')
input_text= st.text_input('enter your topic here')


if input_text:
    st.write(get_llama(input_text))
