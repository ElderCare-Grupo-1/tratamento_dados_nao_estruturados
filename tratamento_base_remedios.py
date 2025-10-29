import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import boto3
from io import StringIO
from dotenv import load_dotenv
import os

if __name__ == '__main__':
    
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
    
    load_dotenv()
    
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_SESSION_TOKEN = os.getenv('AWS_SESSION_TOKEN')
    AWS_REGION = os.getenv('AWS_REGION')

    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_session_token=AWS_SESSION_TOKEN,
        region_name=AWS_REGION
    )

    bucket_name = "s3rawgrupo1"
    file_medicamentos_key = "dados-cleaning/medicamentos/DADOS_ABERTOS_MEDICAMENTOS.csv"
    file_notificacoes_key = "dados-cleaning/medicamentos/VigiMed_Notificacoes.csv"

    # ✅ Ler Medicamentos do S3 → DataFrame
    obj1 = s3.get_object(Bucket=bucket_name, Key=file_medicamentos_key)
    csv1 = obj1['Body'].read().decode('latin1')
    df_base_medicamentos = pd.read_csv(StringIO(csv1), delimiter=';', low_memory=False)

    # ✅ Ler Notificações do S3 → DataFrame
    obj2 = s3.get_object(Bucket=bucket_name, Key=file_notificacoes_key)
    csv2 = obj2['Body'].read().decode('latin1')
    df_base_notificacoes = pd.read_csv(StringIO(csv2), delimiter=';', low_memory=False)

    base_remedios_tratada = (
        df_base_medicamentos['NOME_PRODUTO']
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

    base_notificacoes_tratada = base_notificacoes_tratada.rename(columns={
        'NOME_MEDICAMENTO_WHODRUG': 'NM_MEDICAMENTO',
        'REACAO_EVENTO_ADVERSO_MEDDRA': 'NM_REACAO'
    })

    base_notificacoes_tratada = base_notificacoes_tratada[['NM_MEDICAMENTO', 'NM_REACAO']]
    base_remedios_tratada = base_remedios_tratada.to_frame(name='NM_MEDICAMENTO')
    
    try:
        engine = create_engine("mysql+pymysql://Aluno:Urubu100%40@3.220.74.53:3306/EC_DATA")
        
        with engine.connect() as conn:
            try:
                conn.execute(text("TRUNCATE TABLE EC_DATA.TBL_REACOES_MEDICAMENTOS"))
                conn.execute(text("TRUNCATE TABLE EC_DATA.TBL_MEDICAMENTOS"))
                conn.commit()  # ⚠️ necessário no SQLAlchemy 2.x para comandos DDL/DML
                print("✅ Tabelas truncadas com sucesso!")
            except SQLAlchemyError as e:
                print("❌ Erro ao truncar tabelas:", e)

        chunksize = 20000  # ajustável

        for i, start in enumerate(range(0, len(base_remedios_tratada), chunksize), 1):
            end = start + chunksize
            chunk = base_remedios_tratada.iloc[start:end]

            try:
                chunk.to_sql(
                    name='TBL_MEDICAMENTOS',
                    con=engine,
                    if_exists='append',
                    index=False
                )
                print(f"✅ Chunk {i}: Inseridos {len(chunk)} registros de {start+1} a {min(end, len(base_remedios_tratada))}")
            except SQLAlchemyError as e:
                print(f"❌ Erro no chunk {i}: {e}")

        for i, start in enumerate(range(0, len(base_notificacoes_tratada), chunksize), 1):
            end = start + chunksize
            chunk = base_notificacoes_tratada.iloc[start:end]

            try:
                chunk.to_sql(
                    name='TBL_REACOES_MEDICAMENTOS',
                    con=engine,
                    if_exists='append',
                    index=False
                )
                print(f"✅ Chunk {i}: Inseridos {len(chunk)} registros de {start+1} a {min(end, len(base_notificacoes_tratada))}")
            except SQLAlchemyError as e:
                print(f"❌ Erro no chunk {i}: {e}")
                
                
        print("✅ Inserção concluída com sucesso no MySQL!")

    except SQLAlchemyError as e:
        print("❌ Erro ao inserir no MySQL:", e)
