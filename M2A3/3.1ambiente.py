import os
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama

load_dotenv(find_dotenv())

model_name = os.environ.get("LLM_MODEL")

llm = ChatOllama(model=model_name)

print(llm)