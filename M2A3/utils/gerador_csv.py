import pandas as pd

# Criando o CSV de exemplo com dados de alunos
dados_alunos = pd.DataFrame({
    "nome": ["Ana", "Lucas", "Marina"],
    "curso": ["Administração", "Engenharia", "Pedagogia"],
    "desempenho": ["Baixo", "Alto", "Médio"]
})

# Salvando o CSV para ser usado no exemplo
caminho_csv = "alunos.csv"
dados_alunos.to_csv(caminho_csv, index=False)

caminho_csv
