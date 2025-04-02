from langchain.prompts import PromptTemplate

# Carrega o modelo
def carregar_llm(modelo="mistral"):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=modelo)

# Leitura do arquivo de prompt
with open("prompts/pergunta_tema", "r") as f:
    template = f.read()

# Prompt
prompt = PromptTemplate(
    input_variables=["tema", "curso"],
    template=template
)

# Chain
chain = prompt | carregar_llm()

# Executando o fluxo com as variáveis
resposta = chain.invoke({
    "tema": "Planta",
    "curso": "Biologia"
})

print(resposta.content)