# There is no credit so it will not work
# This code snippet demonstrates how to use the Anthropic chat model with LangChain.


import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Ensure ANTHROPIC_API_KEY is set in your .env file or environment variables
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError("ANTHROPIC_API_KEY not found. Please set it in your .env file or environment.")

model = ChatAnthropic(model_name="claude-2", timeout=30, stop=None)
result = model.invoke("What is the capital of India?")

print("Chat Result:", result)
