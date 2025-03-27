from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Criando o PromptTemplate
from langchain_ollama import ChatOllama

template = "Explique o conceito de {tema} para um estudante do curso de {curso}."
prompt = PromptTemplate(
    input_variables=["tema", "curso"],
    template=template
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0.2
)

# Criando a chain
chain = LLMChain(prompt=prompt, llm=llm)

# Lista de inputs (batch de entradas)
entradas = [
    {"tema": "Big Data", "curso": "Administração"},
    {"tema": "Saúde Básica", "curso": "Enfermagem"},
    {"tema": "Inteligência Artificial", "curso": "Ciência de Dados"}
]

# Executando em lote com .apply()
respostas = chain.apply(entradas)

# Exibindo os resultados
for i, resposta in enumerate(respostas):
    print(f"--- Resposta {i+1} ---")
    print(resposta['text'])
    print()