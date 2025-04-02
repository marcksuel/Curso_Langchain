from langchain.prompts import PromptTemplate

with open("C:/Users/admin/IntelliJ_Workspace/Curso_LangChain/M2A3/prompts/pergunta_tema", "r") as f:
    template = f.read()
    
prompt = PromptTemplate(
    input_variables=["tema", "curso"],
    template=template
)

print(prompt)