from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = [
    "Delhi is capital of India",
    "Mumbai is Capital city of India",
    "Banglore is tech hub",
]

result = embedding.embed_documents(documents)
print(str(result))
