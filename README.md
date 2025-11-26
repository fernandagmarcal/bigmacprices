## 🍔 Análise e Predição do Preço do Big Mac

Este repositório contém o projeto final da disciplina de **Inteligência Artificial**, focado na aplicação e comparação de modelos de **regressão** para prever os preços do sanduíche Big Mac em diferentes países e épocas.

-----

### 🎯 Objetivo do Projeto

O objetivo principal é utilizar o **Machine Learning** para:

1.  Realizar a **Análise Exploratória de Dados (EDA)** no *dataset* de preços do Big Mac.
2.  Implementar e treinar três modelos de regressão distintos.
3.  Avaliar e comparar a performance dos modelos utilizando métricas como **RMSE** e **$R^2$**.
4.  Identificar o modelo mais robusto para a tarefa de predição de preços internacionais.

-----

### 📊 Dataset Utilizado

  * **Nome:** Big Mac Prices
  * **Fonte:** Kaggle
  * **Link:** [https://www.kaggle.com/datasets/vittoriogiatti/bigmacprice?resource=download](https://www.kaggle.com/datasets/vittoriogiatti/bigmacprice?resource=download)
  * **Descrição:** O *dataset* fornece dados históricos de preços do Big Mac em diversas moedas e países, servindo como uma base para o famoso Índice Big Mac (um indicador informal de Paridade do Poder de Compra - PPC).

-----

### 🧠 Modelos de Regressão Implementados

Os seguintes modelos, utilizando a biblioteca **Scikit-learn**, foram treinados e avaliados:

1.  **Regressão Linear (Ridge)**
      * Um modelo linear simples com **regularização L2** (Ridge), que ajuda a prevenir o *overfitting* ao penalizar coeficientes grandes.
2.  **HistGradientBoostingRegressor**
      * Uma implementação de *Gradient Boosting* eficiente e otimizada para conjuntos de dados de médio a grande porte, construindo árvores de decisão aditivamente.
3.  **Random Forest Regressor**
      * Um método de *ensemble* que utiliza a média das predições de múltiplas **Árvores de Decisão** para aumentar a acurácia e estabilidade do resultado.

-----

### 💻 Dependências do Projeto

Para reproduzir a análise e os resultados, você precisará das seguintes bibliotecas Python. Elas podem ser instaladas diretamente através do arquivo `requirements.txt`:

#### `requirements.txt`

```
pandas
scikit-learn
matplotlib
seaborn
joblib
category_encoders
scikit-optimize
```

| Biblioteca | Função no Projeto |
| :--- | :--- |
| **pandas** | Manipulação e análise dos dados (DataFrames). |
| **scikit-learn** | Implementação e avaliação dos modelos de ML. |
| **matplotlib, seaborn** | Visualização de dados e resultados (EDA e gráficos de performance). |
| **joblib** | Serialização dos modelos treinados (salvar/carregar). |
| **category\_encoders** | Pré-processamento de variáveis categóricas (como nomes de países e moedas). |
| **scikit-optimize** | Otimização de hiperparâmetros (como o `alpha` do Ridge ou profundidade do Random Forest). |
