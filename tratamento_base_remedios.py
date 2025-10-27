import pandas as pd

if __name__ == '__main__':
    csv_path = "DADOS_ABERTOS_MEDICAMENTOS.csv"
    csv_notificacoes_path = "VigiMed_Notificacoes.csv"

    lista_frases_remover = [
        'Concentração incorreta de medicamento dispensada',
        'Informação incorreta no prontuário do paciente',
        'Problema de suspeita de qualidade do produto',
        'Medicamento dispensado ao paciente incorreto',
        'Dose incorreta',
        'Administração de dose atrasada',
        'Omissão de dose do produto',
        'Quantidade incorreta de medicamento dispensada',
        'Superdosagem de medicamento',
        'Uso indevido intencional do produto',
        'Odor anormal do produto',
        'Paciente errado',
        'Medicamento ineficaz'
    ]

    df_base = pd.read_csv(csv_path, encoding='latin1', delimiter=';', low_memory=False)
    df_base_notificacoes = pd.read_csv(csv_notificacoes_path, encoding='latin1', delimiter=';', low_memory=False)


    base_remedios_tratada = (
        df_base['NOME_PRODUTO']
        .drop_duplicates()                    # remove duplicatas
        .str.upper()                          # transforma em maiúsculas
        .str.strip()                          # remove espaços no início/fim
        .str.replace(r"[\"']", "", regex=True)  # remove aspas
        .str.replace(r"[^\w]", "", regex=True)  # remove caracteres especiais e espaços
        .reset_index(drop=True)
    )

    base_notificacoes_tratada = (
        df_base_notificacoes[
            ['DATA_ULTIMA_ATUALIZACAO', 'REACAO_EVENTO_ADVERSO_MEDDRA', 'NOME_MEDICAMENTO_WHODRUG']
        ]
        .copy()
    )

    base_notificacoes_tratada['DATA_ULTIMA_ATUALIZACAO'] = (
        pd.to_numeric(base_notificacoes_tratada['DATA_ULTIMA_ATUALIZACAO'], errors='coerce')
        .astype('Int64')
    )

    base_notificacoes_tratada['NOME_MEDICAMENTO_WHODRUG'] = (
        base_notificacoes_tratada['NOME_MEDICAMENTO_WHODRUG']
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"[\"']", "", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    base_notificacoes_tratada = base_notificacoes_tratada.sort_values(
        'DATA_ULTIMA_ATUALIZACAO', ascending=False
    ).reset_index(drop=True)

    mask_erro = base_notificacoes_tratada['REACAO_EVENTO_ADVERSO_MEDDRA'].str.contains('Erro', case=False, na=False)
    mask_concentracao = base_notificacoes_tratada['REACAO_EVENTO_ADVERSO_MEDDRA'].str.contains(
        '|'.join(lista_frases_remover), case=False, na=False
    )
    base_notificacoes_tratada = base_notificacoes_tratada[~(mask_erro | mask_concentracao)]

    base_notificacoes_tratada = base_notificacoes_tratada.drop_duplicates(
        subset=['NOME_MEDICAMENTO_WHODRUG', 'REACAO_EVENTO_ADVERSO_MEDDRA'],
        keep='first'
    ).reset_index(drop=True)

    base_remedios_tratada.to_csv("remedios_base_tratada.csv", index=False, encoding='utf-8-sig')
    base_notificacoes_tratada.to_csv("notificacoes_base_tratada.csv", index=False, encoding='utf-8-sig')

    print("Base de remédios tratada gerada com sucesso!")