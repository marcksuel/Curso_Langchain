from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Modelo
llm = ChatOllama(
    model="llama3.2",
    temperature=0.8
)

# Prompt template
template = "Explique o seguinte texto em linguagem simples: {texto}"
prompt = PromptTemplate(input_variables=["texto"], template=template)

# Chain
chain = prompt | llm

# Execução
entrada = "Aprendizado de máquina é uma subárea da inteligência artificial que permite que sistemas aprendam a partir de dados."
ai_msg = llm.invoke(entrada)
print(ai_msg.content)