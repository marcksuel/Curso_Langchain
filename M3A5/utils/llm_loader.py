from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def load_mistral(template):
    prompt = PromptTemplate.from_template(template)
    llm = ChatOllama(model="mistral")
    chain = prompt | llm
    return chain

def llm_mistral():
    return ChatOllama(model="mistral")
