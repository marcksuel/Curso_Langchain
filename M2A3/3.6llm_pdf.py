from M2A3.utils import llm_loader
from langchain_community.document_loaders import PyPDFLoader #pypdf


def extrair_texto_pdf(caminho_pdf):
    loader = PyPDFLoader(caminho_pdf)
    pages = []
    docs_lazy = loader.lazy_load()
    for page in docs_lazy:
        pages.append(page)
    return pages[0].page_content

# Caminho para o PDF (ajuste conforme necessário)
caminho_pdf = "utils/aluno_exemplo.pdf"  # coloque o nome do seu arquivo aqui

# Extração do texto
texto_pdf = extrair_texto_pdf(caminho_pdf)

# Criando o prompt
template = """
Você é um tutor da universidade.
Crie um plano de estudo com base nas informações do aluno abaixo:

{aluno}
"""

chain = llm_loader.load_mistral(template)

# Executando o modelo com o texto extraído
entrada = {"aluno": texto_pdf}
resposta = chain.invoke(entrada)

print("Resposta da LLM:")
print(resposta.content)
