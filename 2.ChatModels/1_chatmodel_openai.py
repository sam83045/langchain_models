# There is no credit so it will not work
# This code snippet demonstrates how to use the OpenAI chat model with LangChain.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
result = model.invoke("What is the capital of India?")

print("Chat Result:", result)
