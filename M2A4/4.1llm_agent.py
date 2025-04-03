from langchain.agents import Tool, initialize_agent
from langchain_community.llms import Ollama
from M2A3.utils import llm_loader

# Definindo funções que serão usadas como ferramentas

def explicar_conceito(conceito: str) -> str:
    return f"'{conceito}' é um conceito essencial. Recomenda-se começar pelos fundamentos teóricos e exemplos práticos."

def resumir_texto(texto: str) -> str:
    return f"Resumo: {texto}..."

def sugerir_material(tema: str) -> str:
    return f"Sugestão: Consulte o livro 'Introdução a {tema}' da editora Universitária."

# Enumerando as ferramentas
tools = [
    Tool(
        name="Explicador",
        func=explicar_conceito,
        description="Fornece explicações sobre conceitos acadêmicos"
    ),
    Tool(
        name="Resumidor",
        func=resumir_texto,
        description="Resume um texto fornecido pelo usuário"
    ),
    Tool(
        name="Recomendador de Material",
        func=sugerir_material,
        description="Sugere materiais de estudo com base em um tema"
    )
]

# Inicializando o agente com as ferramentas
agent = initialize_agent(
    tools=tools,
    llm=llm_loader.llm_mistral(),
    agent="zero-shot-react-description",
    verbose=True
)

# Executando uma pergunta
resposta = agent.invoke("Preciso de uma explicação sobre o conteúdo acadêmico: Software e Hardware.")
print("Resposta do agente:\n", resposta)  # Resultado gerado pela LLM com base na escolha da ferramenta