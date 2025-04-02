from langchain.prompts import PromptTemplate

with open("prompts/pergunta_tema", "r") as f:
    template = f.read()
    
prompt = PromptTemplate(
    input_variables=["tema", "curso"],
    template=template
)

print(prompt)