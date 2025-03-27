from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

template = """
Você é um tutor acadêmico da universidade. 
Responda à seguinte pergunta de forma clara e objetiva, considerando o perfil do aluno.

Aluno: {aluno}
Pergunta: {pergunta}
"""

prompt = PromptTemplate(
    input_variables=["aluno", "pergunta"],
    template=template
)

# Formatando o prompt com os dados
entrada_formatada = {
    "aluno": "João, aluno do 2º período de Engenharia Elétrica",
    "pergunta": "O que é Internet das Coisas (IoT)?"
}

llm = ChatOllama(
    model="llama3.2",
    temperature=0.2
)

# Chain
chain = prompt | llm

# Executando o fluxo com as variáveis
resposta = chain.invoke(entrada_formatada)

print(resposta.content)