from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from dotenv import load_dotenv
import os

os.environ["HF_HOME"] = "D:/huggingface_cache"


load_dotenv()


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.5, max_new_tokens=100),
)
model = ChatHuggingFace(llm=llm)


st.header("Research tool")
user_input = st.text_input("Enter your prompt")
if st.button("Summarize"):
    result = model.invoke(user_input)
    print(result)
    st.text(result.content)
