# --- Importações de bibliotecas necessárias ---
import streamlit as st  # Biblioteca para criar apps web interativos
import pandas as pd     # Manipulação de dados tabulares
import joblib           # Para carregar modelos e arquivos salvos
import matplotlib.pyplot as plt  # Gráficos
import seaborn as sns             # Visualização estatística
import os                        # Operações com arquivos

# --- Carregar modelos treinados e métricas previamente salvas ---
modelo_rf = joblib.load("modelo_rf.joblib")  # Modelo Random Forest treinado
modelo_dt = joblib.load("modelo_dt.joblib")  # Modelo Decision Tree treinado
metricas_salvas = joblib.load("metricas_modelos.joblib")  # Lista com métricas de avaliação

# --- Função auxiliar para exibir métricas de um modelo ---
def exibir_metricas(modelo):
    # Nome do modelo
    st.markdown(f"### {modelo['modelo']}")
    # Métricas de avaliação
    st.write(f"**Acurácia:** {modelo['acuracia']:.2f}")
    st.write(f"**Precisão:** {modelo['precisao']:.2f}")
    st.write(f"**Recall:** {modelo['recall']:.2f}")
    st.write(f"**F1-score:** {modelo['f1_score']:.2f}")

    # Exibir matriz de confusão com mapa de calor
    st.write("**Matriz de Confusão:**")
    cm = modelo['matriz_confusao']
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Sem Diabetes", "Com Diabetes"],
                yticklabels=["Sem Diabetes", "Com Diabetes"],
                ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    st.pyplot(fig)

# --- Interface com abas laterais: Diagnóstico e Métricas ---
aba = st.sidebar.selectbox("Escolha a visualização:", ["Diagnóstico", "Métricas dos Modelos"])

# --- Aba de Diagnóstico ---
if aba == "Diagnóstico":
    st.title("IA para Diagnóstico Precoce de Diabetes")
    st.markdown("Preencha os dados abaixo:")

    # --- Inputs do usuário (formulário) ---
    # Utilizando widgets do Streamlit para coletar os dados
    idade = st.slider("Idade", 1, 120, 30)
    sexo = st.radio("Sexo:", [0, 1], format_func=lambda x: "Feminino" if x == 0 else "Masculino")
    poliuria = st.radio("Poliúria (urina excessiva)?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    polidipsia = st.radio("Polidipsia (sede excessiva)?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    perda_peso = st.radio("Perda de peso súbita?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    fraqueza = st.radio("Fraqueza?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    polifagia = st.radio("Polifagia (fome excessiva)?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    candidiase = st.radio("Candidíase genital?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    visao_embacada = st.radio("Visão embaçada?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    coceira = st.radio("Coceira?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    irritabilidade = st.radio("Irritabilidade?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    cicatrizacao = st.radio("Cicatrização demorada?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    paresia = st.radio("Paresia parcial?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    rigidez = st.radio("Rigidez muscular?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    alopecia = st.radio("Alopecia?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
    obesidade = st.radio("Obesidade?", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")

    # --- Montar DataFrame com os dados preenchidos ---
    nova_pessoa = pd.DataFrame([{
        'Idade': idade,
        'Sexo': sexo,
        'Poliúria': poliuria,
        'Polidipsia': polidipsia,
        'Perda de Peso Súbita': perda_peso,
        'Fraqueza': fraqueza,
        'Polifagia': polifagia,
        'Candidíase Genital': candidiase,
        'Visão Embaçada': visao_embacada,
        'Coceira': coceira,
        'Irritabilidade': irritabilidade,
        'Cicatrização Demorada': cicatrizacao,
        'Paresia Parcial': paresia,
        'Rigidez Muscular': rigidez,
        'Alopecia': alopecia,
        'Obesidade': obesidade
    }])

    # --- Botão para gerar a predição ---
    if st.button("Analisar Risco de Diabetes"):
        # Previsões com os modelos carregados
        pred_rf = modelo_rf.predict(nova_pessoa)[0]
        prob_rf = modelo_rf.predict_proba(nova_pessoa)[0][1]

        pred_dt = modelo_dt.predict(nova_pessoa)[0]
        prob_dt = modelo_dt.predict_proba(nova_pessoa)[0][1]

        st.subheader("Resultados da Predição")

        # Resultado Random Forest
        if pred_rf == 1:
            st.error(f"\U0001F9E0 Random Forest: Diabetes com {round(prob_rf * 100, 2)}% de chance.")
        else:
            st.success(f"\u2705 Random Forest: Sem Diabetes com {round((1 - prob_rf) * 100, 2)}% de confiança.")

        # Resultado Decision Tree
        if pred_dt == 1:
            st.error(f"\U0001F52C Árvore de Decisão: Diabetes com {round(prob_dt * 100, 2)}% de chance.")
        else:
            st.success(f"\u2705 Árvore de Decisão: Sem Diabetes com {round((1 - prob_dt) * 100, 2)}% de confiança.")

        # --- Salvar entrada com resultado ---
        nova_pessoa['Diabetes_RF'] = pred_rf
        nova_pessoa['Diabetes_DT'] = pred_dt

        # Salvar ou atualizar arquivo CSV
        if os.path.exists("entradas.csv"):
            entradas = pd.read_csv("entradas.csv")
            entradas = pd.concat([entradas, nova_pessoa], ignore_index=True)
        else:
            entradas = nova_pessoa

        entradas.to_csv("entradas.csv", index=False)

        # --- Gráfico com dados acumulados de diagnósticos ---
        st.subheader("Diagnósticos Acumulados")
        grafico_data = entradas["Diabetes_RF"].value_counts().reindex([0, 1], fill_value=0)
        grafico_data.index = ["Sem Diabetes", "Com Diabetes"]
        st.bar_chart(grafico_data)

# --- Aba de Métricas dos Modelos ---
elif aba == "Métricas dos Modelos":
    st.title("Desempenho dos Modelos")
    for metrica in metricas_salvas:
        exibir_metricas(metrica)
