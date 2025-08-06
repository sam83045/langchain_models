from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np


load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Cricketers are professional athletes who play the sport of cricket, representing their countries or clubs in various international and domestic competitions.",
    "Some of the most famous and influential cricketers as of 2025 include Virat Kohli, Sachin Tendulkar, Rohit Sharma, Babar Azam, and MS Dhoni.",
    "Many top cricketers hold extraordinary records, such as Sachin Tendulkar’s 100 international centuries and Brian Lara’s highest individual Test score of 400 not out.",
    "Modern cricketers often enjoy massive global followings, with millions of fans on social media—Virat Kohli alone boasts over 270 million Instagram followers, making him one of the most popular athletes in the world.",
    "Beyond the pitch, legendary cricketers contribute to coaching, mentoring, philanthropy, and promoting the game worldwide.",
]

doc_embedding = embedding.embed_documents(documents)

query = "Who is having many followers"
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print(query)
print(documents[index])
print("Similarity Score is ", score)
