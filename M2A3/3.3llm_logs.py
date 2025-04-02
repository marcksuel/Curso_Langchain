from langchain.prompts import PromptTemplate


# Carrega o modelo
def carregar_llm(modelo="mistral"):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=modelo)

def log(resposta):
    with open("logs.txt", "a", encoding="utf-8") as log:
        log.write(f"[Resposta]: {resposta}\n")

# Assistente
def criar_assistente_tutor(llm):
    with open("prompts/prompt_universidade", "r") as f:
        template = f.read()
    prompt = PromptTemplate.from_template(template)
    return prompt | llm

chain = criar_assistente_tutor(carregar_llm("mistral"))

resposta = chain.invoke({
    "pergunta": "Como melhorar em cálculo?",
    "curso": "Engenharia"
})

log(resposta)

print(resposta.content)
