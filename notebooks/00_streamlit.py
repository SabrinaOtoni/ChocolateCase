# Rodar os comandos "cd notebooks" e depois "streamlit run 00_streamlit.py" no terminal. Para parar a execução, clicarl em "stop" na tela do Streamlit.

import streamlit as st
import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe
from pathlib import Path
import joblib

rf_model = joblib.load(Path('../model') / 'rf_model.pkl')
preprocessor = joblib.load(Path('../preprocessing') / 'preprocessor_hot.pkl')

def cost_function(p):
    return 200 / (1 + np.exp(10 * (p - 9.5))) + 200 / (1 + np.exp(-0.8 * (p - 12)))

def optimize_qtd_choc(row):
    space = hp.uniform('QTD_CHOC', 110, 440)

    def objective(qtd_choc):
        predicted_weight = rf_model.predict(np.array([[qtd_choc, row['VAR_1'], row['VAR_2_B'], row['VAR_2_C']]]))[0]
        cost = cost_function(predicted_weight)
        return cost, predicted_weight

    def objective_wrapper(qtd_choc):
        return objective(qtd_choc)[0]

    best = fmin(
        fn=objective_wrapper,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        rstate=np.random.default_rng(0)
    )
    _, predicted_weight = objective(best['QTD_CHOC'])
    return best['QTD_CHOC'], predicted_weight

def verificar_colunas(dataframe, colunas_requeridas):
    colunas_faltando = [coluna for coluna in colunas_requeridas if coluna not in dataframe.columns]
    return colunas_faltando

st.title('Otimizador de Quantidade de Chocolate')

uploaded_file = st.file_uploader("Escolha um arquivo XLSX", type="xlsx")

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        data_prescriptive = pd.read_excel(uploaded_file)
        colunas_requeridas = ['QTD_CHOC', 'VAR_1', 'VAR_2']
        colunas_faltando = verificar_colunas(data_prescriptive, colunas_requeridas)

        if len(colunas_faltando) > 0:
            st.error(f"Faltam as seguintes colunas no arquivo: {', '.join(colunas_faltando)}")
        else:
            data_prescriptive = pd.read_excel(uploaded_file)

            data_prescriptive_hot_encoded = pd.DataFrame(preprocessor.transform(data_prescriptive), columns=preprocessor.get_feature_names_out())
            data_prescriptive_hot_encoded.columns = [col.split("__")[-1] for col in data_prescriptive_hot_encoded.columns]
            data_prescriptive_hot_encoded = pd.concat([data_prescriptive, data_prescriptive_hot_encoded], axis=1)
            data_prescriptive_hot_encoded.drop('VAR_2', axis=1, inplace=True)
    
            results = data_prescriptive_hot_encoded.apply(lambda row: optimize_qtd_choc(row) if pd.isnull(row['QTD_CHOC']) else (row['QTD_CHOC'], np.nan), axis=1)

            data_prescriptive_hot_encoded['QTD_CHOC'] = results.apply(lambda x: x[0])
            data_prescriptive_hot_encoded['PESO_BOMBOM_PREV'] = results.apply(lambda x: x[1])

            st.write(data_prescriptive_hot_encoded.describe())

            csv = data_prescriptive_hot_encoded.to_csv(index=False)
            st.download_button("Baixar dados", csv, "data_prescriptive.csv", "text/csv", key='download-csv')
    else:
        st.error("O arquivo carregado não é um arquivo Excel (.xlsx). Por favor, carregue um arquivo Excel válido.")