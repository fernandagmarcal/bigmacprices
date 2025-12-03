import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================
st.set_page_config(
    page_title="Big Mac Index & Analytics",
    page_icon="🍔",
    layout="wide"
)

# Estilo Personalizado
st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .stButton>button { background-color: #FFC72C; color: black; font-weight: bold; width: 100%; border-radius: 8px;}
    </style>
    """, unsafe_allow_html=True)

st.title("🍔 Big Mac: Analytics & Previsão")
st.markdown("### Inteligência Artificial para prever a inflação global do Big Mac")

# ==================================================
# 1. CARREGAMENTO DO MODELO
# ==================================================
@st.cache_resource
def load_model():
    if not os.path.exists("modelo_bigmac.pkl"):
        st.error("❌ Arquivo 'modelo_bigmac.pkl' não encontrado.")
        return None
    try:
        return joblib.load("modelo_bigmac.pkl")
    except Exception as e:
        st.error(f"❌ Erro ao ler arquivo: {e}")
        st.warning("Se o erro for de versão, tente atualizar o scikit-learn no seu PC: pip install --upgrade scikit-learn")
        return None

dados = load_model()
if dados is None: st.stop()

model = dados['modelo']
preprocessor = dados['preprocessor']

# ==================================================
# 2. INPUTS (BARRA LATERAL)
# ==================================================
st.sidebar.header("⚙️ Configuração")

# Recupera lista de países do processador (extraindo do OneHotEncoder)
try:
    # Tenta pegar as categorias aprendidas pelo OneHotEncoder
    # O Pipeline geralmente nomeia os passos. O nosso chamava 'cat' dentro do ColumnTransformer
    lista_paises = list(preprocessor.named_transformers_['cat'].categories_[0])
    lista_paises.sort()
except:
    # Fallback caso a estrutura do pickle seja diferente
    lista_paises = ["Argentina", "Australia", "Brazil", "Britain", "Canada", "China", "Euro area", "Japan", "Switzerland", "United States"]

# Inputs
index_brasil = lista_paises.index("Brazil") if "Brazil" in lista_paises else 0
pais_selecionado = st.sidebar.selectbox("Escolha um País", lista_paises, index=index_brasil)

# Data input com formato DD/MM/YYYY
data_alvo = st.sidebar.date_input("Data da Previsão", datetime.now(), format="DD/MM/YYYY")

calcular = st.sidebar.button("CALCULAR PREVISÕES 🚀")

st.sidebar.markdown("---")
st.sidebar.info("Este modelo utiliza Regressão Linear para tendências de longo prazo e Árvores de Decisão para padrões locais.")

# ==================================================
# 3. DASHBOARD PRINCIPAL
# ==================================================

if calcular:
    # --- A. PREVISÃO INDIVIDUAL ---
    # Cria DF simples (o preprocessor vai transformar em One-Hot sozinho)
    entrada_usuario = pd.DataFrame({
        'name': [pais_selecionado], 
        'year': [data_alvo.year], 
        'month': [data_alvo.month]
    })
    
    try:
        # Transforma e Preve
        entrada_proc = preprocessor.transform(entrada_usuario)
        preco_usuario = model.predict(entrada_proc)[0]
        
        # Cria DF para os EUA (Benchmark) na mesma data
        entrada_usa = pd.DataFrame({'name': ['United States'], 'year': [data_alvo.year], 'month': [data_alvo.month]})
        preco_usa = model.predict(preprocessor.transform(entrada_usa))[0]
        
        # Exibe KPIs no topo
        kpi1, kpi2, kpi3 = st.columns(3)
        
        kpi1.metric(
            label=f"Preço em {pais_selecionado}", 
            value=f"US$ {preco_usuario:.2f}",
            delta=f"{((preco_usuario/preco_usa)-1)*100:.1f}% vs EUA"
        )
        
        kpi2.metric(
            label="Preço nos EUA (Referência)", 
            value=f"US$ {preco_usa:.2f}"
        )
        
        cotacao_estimada = 5.00 
        kpi3.metric(
            label="Estimativa em Reais (R$)", 
            value=f"R$ {preco_usuario * cotacao_estimada:.2f}",
            help="Cotação fixa de exemplo: US$ 1 = R$ 5.00"
        )
        
        st.divider()

        # --- B. RANKING GLOBAL ---
        st.subheader(f"🌍 Ranking Mundial Estimado para {data_alvo.year}")
        
        # 1. Cria um DataFrame com TODOS os países para aquela data
        df_global = pd.DataFrame({
            'name': lista_paises,
            'year': [data_alvo.year] * len(lista_paises),
            'month': [data_alvo.month] * len(lista_paises)
        })
        
        # 2. Preve tudo de uma vez
        precos_globais = model.predict(preprocessor.transform(df_global))
        df_global['Preço Estimado (USD)'] = precos_globais
        
        # 3. Ordena
        df_caros = df_global.sort_values('Preço Estimado (USD)', ascending=False).head(10)
        df_baratos = df_global.sort_values('Preço Estimado (USD)', ascending=True).head(10)
        
        # 4. Gráficos lado a lado
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.markdown("**💰 Top 10 Mais Caros**")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df_caros, x='Preço Estimado (USD)', y='name', palette='Reds_r', ax=ax1)
            ax1.set_xlabel("Preço (USD)")
            ax1.set_ylabel("")
            st.pyplot(fig1)
            
        with col_graf2:
            st.markdown("**🏷️ Top 10 Mais Baratos**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df_baratos, x='Preço Estimado (USD)', y='name', palette='Greens_r', ax=ax2)
            ax2.set_xlabel("Preço (USD)")
            ax2.set_ylabel("")
            st.pyplot(fig2)

        # --- C. GRÁFICO DE TENDÊNCIA COMPARATIVA ---
        st.divider()
        st.subheader("📈 Curva de Inflação: Selecionado vs EUA")
        
        anos = list(range(2000, data_alvo.year + 6))
        
        dados_pais = []
        dados_usa = []
        
        # Previsão em batch para ser mais rápido
        batch_pais = pd.DataFrame({'name': [pais_selecionado]*len(anos), 'year': anos, 'month': [7]*len(anos)})
        batch_usa = pd.DataFrame({'name': ['United States']*len(anos), 'year': anos, 'month': [7]*len(anos)})
        
        preds_pais = model.predict(preprocessor.transform(batch_pais))
        preds_usa = model.predict(preprocessor.transform(batch_usa))
            
        # Plot
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(anos, preds_pais, label=pais_selecionado, color='#DA291C', linewidth=3, marker='o')
        ax3.plot(anos, preds_usa, label="United States", color='#003399', linewidth=2, linestyle='--')
        
        # Marca o ponto atual
        ax3.axvline(x=data_alvo.year, color='gray', linestyle=':', alpha=0.5)
        
        ax3.set_ylabel("Preço Big Mac (USD)")
        ax3.set_title(f"Comparativo de Evolução de Preço")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        st.pyplot(fig3)
        
    except Exception as e:
        st.error(f"Erro no cálculo: {e}")

else:
    # Tela inicial
    st.info("👈 Selecione um país e uma data na barra lateral para começar a análise.")