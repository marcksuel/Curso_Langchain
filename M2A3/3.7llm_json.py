from M2A3.utils import llm_loader
import json

# Carregar dados do JSON
with open("utils/feedback.json", "r", encoding="utf-8") as f:
    entrada = json.load(f)

# Criando o prompt
template = """
Você é um tutor universitário.

Com base nas informações abaixo, escreva um feedback construtivo para o aluno, oferecendo sugestões de estudo:

Aluno: {nome}  
Curso: {curso}  
Comentário: {comentario}
"""

chain = llm_loader.load_mistral(template)

# Executando
resposta = chain.invoke(entrada)

print("Resposta da LLM:")
print(resposta.content)
