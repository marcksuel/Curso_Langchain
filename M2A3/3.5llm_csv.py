import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

df = pd.read_csv("utils/alunos.csv")  # Certifique-se de que o arquivo esteja no mesmo diretório
linha = df.iloc[0]

# Criar lista de entradas (um dicionário por aluno)
entradas = df.to_dict(orient="records")
entrada_unica = linha.to_dict()

# Prompt
template = """
Gere um feedback personalizado com base nos dados abaixo:

Nome: {nome}  
Curso: {curso}  
Desempenho: {desempenho}
"""
prompt = PromptTemplate.from_template(template)

# Carregar LLM
llm = ChatOllama(model="mistral")

# Chain
chain = prompt | llm

# Executar uma entrada com .invoke()
resposta = chain.invoke(entrada_unica)
print(resposta.content)


# # Executar em lote com .batch()
# respostas = chain.batch(entradas)
#
# # Resultados
# for i, resposta in enumerate(respostas):
#     print(f"Aluno {entradas[i]['nome']}:\n{resposta.content}\n{'-'*50}")