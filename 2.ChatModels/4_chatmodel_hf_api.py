from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm= HuggingFaceEndpoint(    
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

model= ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print("Chat Result:", result)