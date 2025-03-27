from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

template = "Explique brevemente o que é {tema} para um estudante do curso de {curso}"

prompt = PromptTemplate(
    input_variables=["tema", "curso"],
    template=template
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0.2
)

#Ajuda no debugging
#prompt.format(tema="Big Data", curso="Administração")
#print(prompt)

# Chain
chain = prompt | llm

# Executando o fluxo com as variáveis
# resposta = chain.invoke({
#     "tema": "Planta",
#     "curso": "Biologia"
# })

resposta = chain.invoke({
    "tema": "Planta",
    "curso": "Arquitetura"
})

print(resposta.content)