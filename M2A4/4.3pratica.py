from langchain_community.tools import ArxivQueryRun, YouTubeSearchTool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from M2A3.utils import llm_loader

# Configuração das ferramentas
arxiv = ArxivQueryRun(
    description="Ferramenta para buscar artigos acadêmicos e papers. Use para conceitos técnicos ou científicos."
)

wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia = WikipediaQueryRun(
    description="Ferramenta para explicações textuais de conceitos. Use para definições e contextos históricos.",
    api_wrapper=wiki_api_wrapper
)

youtube = YouTubeSearchTool(
    description="Ferramenta para buscar vídeos explicativos. Use quando conceitos são melhor aprendidos visualmente."
)

tools = [wikipedia, arxiv, youtube]

# Carregar o modelo LLM
chat_model = llm_loader.llm_mistral()

# Prompt do sistema especializado
system_prompt = SystemMessage("""
Você é um assistente especializado em preparação para provas chamado ProMaster. Sua função é:

1. Analisar o tema fornecido pelo aluno
3. Propor uma prova que deve conter:
   - Definições conceituais
   - 5 questões no maximo
   - 2 ou mais questões de multipla escolha com opções 5
   - 1 ou mais questões de verdadeiro ou falso

Sempre priorize fontes confiáveis e adapte o prova ao nível do aluno.
""")

# Criar o agente
agent = create_react_agent(chat_model, tools, state_modifier=system_prompt)


def gerar_plano_estudo(tema: str):
    """Função para invocar o agente e gerar o plano de estudos"""
    response = agent.invoke({"messages": [
        HumanMessage(f"""
        Preciso de uma prova sobre: {tema}

        Por favor, gere uma prova completo com:
        1. Os conceitos-chave que devem ser dominados
        2. Apenas questões
        3. Caso seja necessario questões de calculo, utilizar multipla escolha
        """)
    ]})

    print("\n" + "=" * 50)
    print(f"PROVA PARA O TEMA: {tema.upper()}")
    print("=" * 50 + "\n")

    # Processar a resposta
    for message in response['messages']:
        if isinstance(message, HumanMessage):
            continue  # Ignorar a mensagem do usuário na saída

        print(f"{message.__class__.__name__}:")
        print(message.content)
        print("-" * 50)


# Interface com o usuário
if __name__ == "__main__":
    while(True):
        print("Bem-vindo ao Assistente de Preparação para Provas!")
        tema = input("Digite o tema principal da sua prova: ")
        gerar_plano_estudo(tema)