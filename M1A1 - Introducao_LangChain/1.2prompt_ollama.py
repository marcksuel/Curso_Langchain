from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain

llm = ChatOllama(
    # model="mistral",
    model="llama3.2",
    temperature=0.9
)

template = "Resuma o seguinte texto em poucas palavras: {conteudo}"
prompt = PromptTemplate(input_variables=["conteudo"], template=template)


chain = LLMChain(llm=llm, prompt=prompt)
saida = chain.run(conteudo="O Aprendizado de Máquina é uma área específica da Inteligência Artificial (IA) que permite "
                           "a criação de algoritmos capazes de aprender e melhorar suas performances com base em dados. "
                           "Esse conhecimento pode ser aplicado em diversas áreas, como ciência de dados, engenharia, "
                           "medicina, entre outras, tornando-o uma ferramenta poderosa para otimizar processos e decisões.")
print(saida)
