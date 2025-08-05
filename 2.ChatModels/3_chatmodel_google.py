# This code snippet demonstrates how to use the Google chat model with LangChain.
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv  

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
result = model.invoke("What is the capital of India?")

print("Chat Result:", result.content)
