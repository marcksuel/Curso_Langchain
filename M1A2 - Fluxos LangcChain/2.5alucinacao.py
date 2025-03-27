from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Modelo
llm = ChatOllama(
    model="llama3.2",
    temperature=0.8
)

ai_msg = llm.invoke("Já houve um senador dos EUA que serviu o estado de Minnesota e cuja amor foi a Universidade de Princeton?")
# ai_msg = llm.invoke("Hubert Horatio Humphrey Jr foi a Universidade de Princeton?")
print(ai_msg.content)