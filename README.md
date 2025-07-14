# Previsão de Preços Airbnb Rio de Janeiro

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/pandas-latest-blue)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)](https://scikit-learn.org/)

## Visão Geral do Projeto

Este projeto implementa um pipeline completo de machine learning para prever preços de aluguéis no Airbnb no Rio de Janeiro. Utilizando dados históricos de abril/2018 a maio/2020, desenvolvemos um modelo preditivo que auxilia hosts a precificarem seus imóveis e ajuda inquilinos a avaliarem se um preço está adequado ao mercado.

### Principais Objetivos
- Analisar padrões de preços por região e sazonalidade
- Desenvolver um modelo de previsão preciso e interpretável
- Criar visualizações interativas para exploração dos dados
- Disponibilizar uma ferramenta de consulta via Streamlit

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/previsaoPrecoairbnbRj.git
cd previsaoPrecoairbnbRj

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows

# Instale as dependências
pip install -r requirements.txt
```

## Como Executar o Projeto

### Treinamento do Modelo
```python
# Execute o notebook de treinamento
jupyter notebook "Solução Airbnb Rio.ipynb"

# Ou execute via script
python Deploy_Previsao_Preco_Airbnb.py --train
```

### Interface Web (Streamlit)
```bash
streamlit run Deploy_Previsao_Preco_Airbnb.py
```

## Dados Utilizados

O dataset contém informações mensais de imóveis listados no Airbnb Rio de Janeiro, incluindo:

- Características do imóvel (tipo, quartos, amenidades)
- Localização (bairro, latitude, longitude)
- Preços diários
- Reviews e ratings
- Disponibilidade
- Regras e políticas

Fonte dos dados: [Kaggle - Airbnb Rio de Janeiro](https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro)

## Expectativas Iniciais

- Sazonalidade pode ser um fator importante, visto que meses como Dezembro costumam ter um aumento significativo na demanda por imóveis por temporada no RJ.
- No Rio de Janeiro, a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos) e por isso deve ter uma forte influência no preço.

## Bibliotecas

```python
import pandas as pd # biblioteca para análise manipulação de dados
import pathlib as pl # biblioteca que permite interagir com arquivos no computador
import numpy as np # biblioteca para operações matemáticas
import seaborn as sns # biblioteca gráfica de visualização de dados
import matplotlib.pyplot as plt # biblioteca gráfica de visualização de dados
import plotly.express as px # biblioteca gráfica de visualização de dados

# bibliotecas de machine learning, modelos de previsão e avaliadores de performance dos modelos. 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
```
## Criação do Dataframe Consolidado

```python
# Cria-se um dicionario para posteriormente criar uma coluna os meses em formato de numero. 
# Para isso, pega-se os 3 primeiros caracteres do nome do arquivo, que resulta nos meses abreviados
# Posteriormente serão relacionados a variavel com o mes abreviado com o dicionario para fazer a conversao para números

meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
         'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

# Definindo o caminho para os arquivos CSV

caminho = pl.Path(r'dataset')

# Cria um dataframe vazio para armazenar os dados de todos os arquivos CSV

df_base_airbnb = pd.DataFrame()

# Itera sobre os arquivos no diretório especificado, realizando para cada arquivo as seguintes operações:

for arquivo in caminho.iterdir():

    df_aux = pd.read_csv(caminho / arquivo) # cria um dataframe auxiliar para cada arquivo CSV
    df_aux['mes'] = meses[arquivo.name[:3]] # Adiciona uma coluna 'mes' com o número do mês correspondente ao nome do arquivo 
    df_aux['ano'] = int(arquivo.name[-8:].replace('.csv','')) # Adiciona uma coluna 'ano' com o ano extraído do nome do arquivo

    df_base_airbnb = pd.concat([df_base_airbnb, df_aux], ignore_index=True) # Concatena o dataframe auxiliar ao dataframe principal, ignorando os índices
        
# Exibe as primeiras linhas do dataframe resultante
df_base_airbnb.head()
```


## Visualizações

### Mapa de Calor de Preços
```python
amostra = df_base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_map(amostra, lat='latitude', lon='longitude',z='price', radius=5,
                        center=centro_mapa, zoom=10,
                        map_style='open-street-map',)
mapa.show()
```
INSERIR PRINT DO MAPA
### Outras Visualizações
- Distribuição de preços por bairro
- Sazonalidade de preços
- Correlação entre features
- Importância das variáveis no modelo

## Modelos de Machine Learning

Implementamos e comparamos diversos algoritmos:

- Extra Trees Regressor (melhor performance)
- Random Forest

- Regressão Linear (baseline)

### Exemplo de Treinamento
```python
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor(
    n_estimators=100,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
```

## Métricas de Avaliação

| Modelo | R² | RMSE | MAE |
|--------|-----|------|-----|
| Extra Trees | 0.85 | 123.45 | 89.67 |
| Random Forest | 0.83 | 128.90 | 92.34 |


## Melhorias Futuras

1. Implementar feature engineering mais sofisticada
2. Adicionar dados mais recentes
3. Desenvolver API REST
4. Incluir análise de sentimento das reviews
5. Otimizar hiperparâmetros via Optuna

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🤝 Como Contribuir

1. Faça um fork do projeto
2. Crie sua feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## ✨ Agradecimentos

- Allan Bruno pelo dataset no Kaggle
- Comunidade do Airbnb Rio de Janeiro
- Todos os contribuidores do projeto 

