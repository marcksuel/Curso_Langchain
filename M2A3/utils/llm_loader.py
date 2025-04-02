from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

def load_mistral(template):
    prompt = PromptTemplate.from_template(template)

    # Carregar LLM
    llm = ChatOllama(model="mistral")

    # Chain
    chain = prompt | llm
    
    return chain

def llm_mistral():
    # Carregar LLM
    llm = ChatOllama(model="mistral")
    return llm