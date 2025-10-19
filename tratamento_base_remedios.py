import pandas as pd

# Caminho da base original
csv_path = "DADOS_ABERTOS_MEDICAMENTOS.csv"

# ---------- Passo 1: Ler a base ----------
df_base = pd.read_csv(csv_path, encoding='latin1', delimiter=';')

# ---------- Passo 2: Extrair somente NOME_PRODUTO e padronizar ----------
base_remedios_tratada = (
    df_base['NOME_PRODUTO']
    .drop_duplicates()                    # remove duplicatas
    .str.upper()                          # transforma em maiúsculas
    .str.strip()                          # remove espaços no início/fim
    .str.replace(r"[\"']", "", regex=True)  # remove aspas
    .str.replace(r"[^\w]", "", regex=True)  # remove caracteres especiais e espaços
    .reset_index(drop=True)
)

# ---------- Passo 3: Salvar base tratada ----------
base_remedios_tratada.to_csv("remedios_base_tratada.csv", index=False, encoding='utf-8-sig')

print("Base de remédios tratada gerada com sucesso!")