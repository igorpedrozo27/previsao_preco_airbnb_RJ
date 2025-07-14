import pandas as pd # biblioteca para análise manipulação de dados
import pathlib as pl # biblioteca que permite percorrer arquivos no computador
import numpy as np # biblioteca para arrays e operações matemáticas
import seaborn as sns # biblioteca gráfica de visualização de dados
import matplotlib.pyplot as plt # biblioteca gráfica de visualização de dados
import plotly.express as px # biblioteca gráfica de visualização de dados

# bibliotecas de machine learning, modelos de previsão e avaliadores de performance dos modelos. 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# Cria-se um dicionario para criar uma coluna os meses em formato de numero. 
# Para isso, pega-se os 3 primeiros caracteres do nome do arquivo, que resulta nos meses abreviados
# Posteriormente serão relacionados a variavel com o mes abreviado com o dicionario para fazer a conversao para números

meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
         'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

# Definindo o caminho para os arquivos CSV

caminho = pl.Path(r'C:/Users/igor_/Programacao/Cursos Hashtag/Python/Projeto de Ciencia de Dados e Recomendacoes/dataset/dataset/')

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

# ### Consolidar Base de Dados para Analise Qualitativa
# Será criada e exportada para csv uma amostra da base de dados completa para que seja feita uma analise e exclusão de features(colunas) desnecessárias para o objetivo final do projeto no Excel.

df_base_airbnb.head(1000).to_csv('base_airbnb_primeiros_registros.csv', index=False, sep=',', encoding='utf-8-sig')

# Colunas em que todos ou quase todos os valores são iguais - 'host_listings_count' e 'host_total_listings_count'

# Compara a coluna 'host_listings_count' com 'host_total_listings_count' e imprime o resultado
print((df_base_airbnb['host_listings_count']==df_base_airbnb['host_total_listings_count']).value_counts())

# Verifica e imprime quantas entradas na coluna 'square_feet' são nulas

print(df_base_airbnb['square_feet'].isnull().sum())

# Após análise feita no Excel com o arquivo exportado,
# 
# - Tipos de colunas excluídas:
#     1. IDs, Links e informações não relevantes para o modelo
#     2. Colunas que fornecem informações similares (Ex: Data x Ano/Mês)

# ### Filtrando a Base de dados após Analise Qualitativa - exclusão de colunas feita diretamente no Excel.

colunas_remanescentes = 'host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','mes','ano'

# Seleciona apenas as colunas remanescentes do dataframe df_base_airbnb

df_base_airbnb = df_base_airbnb.loc[:, colunas_remanescentes]
print(df_base_airbnb)

# ### Tratamento de Valores Faltantes
print(df_base_airbnb.isnull().sum())

# percorre a base de dados e exclui colunas que possuem mais de 100.000 valores nulos. 
for coluna in df_base_airbnb:
    if (df_base_airbnb[coluna].isnull().sum()) > 100000:
        df_base_airbnb.drop(columns=coluna, axis=1, inplace=True)
        print(f'Coluna {coluna} excluida')

print('=============================================')
print(df_base_airbnb.isnull().sum())
# ### Removendo as linhas com dados nulos

print(df_base_airbnb.shape)

linhas_ant = df_base_airbnb.shape[0] # coloca em uma variavel o valor da qtd de linhas da tupla retornada pelo .shape

df_base_airbnb = df_base_airbnb.dropna() # remove as linhas que possuem valores nulos em qualquer coluna do dataframe

linhas_post = df_base_airbnb.shape[0] # coloca em uma variavel o valor da qtd de linhas da tupla retornada pelo .shape, 

print(f'Novo shape do dataframe (linhas, colunas): {df_base_airbnb.shape}')

print(f'{linhas_ant - linhas_post} linhas foram removidas do dataframe')

# OBS: O dropna() remove todas as linhas que possuem pelo menos um valor nulo, se quiser remover apenas as linhas que possuem valores nulos em todas as colunas, deve-se usar o parâmetro how='all' no método dropna()

print(df_base_airbnb.isnull().sum())
# ### Verificando os Tipos de Dados em cada coluna
print(df_base_airbnb.info())

print(df_base_airbnb.dtypes)
print('='*100)

print(df_base_airbnb.loc[0,:])
# ### Alteração de tipo de dado das variáveis que são object e precisam ser float
# Remove '$' e ',' e converte para float
df_base_airbnb['price'] = df_base_airbnb['price'].str.replace('[$,]', '', regex=True).astype(np.float32, copy=False)

df_base_airbnb['extra_people'] = df_base_airbnb['extra_people'].str.replace('[$,]', '', regex=True).astype(np.float32, copy=False)

print(df_base_airbnb.dtypes)

# ### Análise Exploratória e Tratamento de Outliers
# ##### Plotando a matriz de correlação das features

plt.figure(figsize=(15,5))
sns.heatmap(df_base_airbnb.corr(numeric_only=True), annot=True, cmap='coolwarm')

# ##### - Para o tratamento de outliers iremos aplicar o método do <b>Intervalo Interquartil</b> de Estatística Descritiva 

# O método do Intervalo Interquartil (IQR – Interquartile Range) é uma técnica da estatística descritiva muito usada para:
# 
# - Resumir a dispersão dos dados (variabilidade)
# - Identificar outliers (valores discrepantes)
# 
# O IQR é a diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1) de um conjunto de dados ordenado:
# 
# - Intervalo Interquartil: 
#     - <b>IQR = Q3 − Q1</b>
# 
#         - Q1 (1º quartil): valor abaixo do qual estão 25% dos dados
#         - Q3 (3º quartil): valor abaixo do qual estão 75% dos dados
#         - IQR: representa a "faixa do meio" dos 50% centrais da amostra
# 
# Valores são considerados outliers se estiverem muito distantes do centro dos dados.
# 
# - Limite inferior: 
#     - <b>LI = Q1 − 1.5 × IQR</b>
# - Limite superior: 
#     - <b>LS= Q3 + 1.5 × IQR</b>
# 
# Qualquer valor fora desses limites é um outlier.

# ### Definindo funções de definição de limite inferior e limite superior para analise de cada feature (coluna)

def limites (coluna):
    """
    Calcula os limites inferior e superior para a coluna especificada.
    Utiliza o método IQR (Intervalo Interquartil) para detectar outliers.
    """
    Q1 = df_base_airbnb[coluna].quantile(0.25)
    Q3 = df_base_airbnb[coluna].quantile(0.75)
    IQR = Q3 - Q1 # amplitude do intervalo interquartil
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return float(limite_inferior), float(limite_superior)

def remove_outliers(coluna):
    '''Esta função remove os outliers da coluna especificada. Utiliza os limites calculados pela função limites().'''
    
    qtd_linhas_inicial = df_base_airbnb.shape[0] # pega a quantidade de linhas do dataframe antes da remoção dos outliers
    limite_inferior, limite_superior = limites(coluna) # calcula os limites inferior e superior para a coluna especificada
    df_base_airbnb.drop(df_base_airbnb[(df_base_airbnb[coluna] < limite_inferior) | (df_base_airbnb[coluna] > limite_superior)].index, inplace=True)
    # Atualiza o dataframe removendo as linhas que estão fora dos limites calculados
    qtd_linhas_final = df_base_airbnb.shape[0] # calcula a quantidade de linhas do dataframe após a remoção dos outliers
    linhas_removidas = qtd_linhas_inicial - qtd_linhas_final # calcula a quantidade de linhas removidas
    return df_base_airbnb, linhas_removidas

# ### Definindo função para plotar e analisar os graficos de cada feature (coluna)

def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.boxplot(x=df_base_airbnb[coluna], ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=df_base_airbnb[coluna], ax=ax2)
    """
    Plota um diagrama de caixa para a coluna especificada.
    """  

def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.histplot(df_base_airbnb[coluna], kde=True, bins=50)
    """
    Plota um histograma para a coluna especificada.
    """

def barras(coluna):
    plt.figure(figsize=(15, 5))
    sns.barplot(x=df_base_airbnb[coluna].value_counts().index, y=df_base_airbnb[coluna].value_counts())
    """
    Plota um gráfico de barras para a coluna especificada.
    """
# ## Analise de Valores Numéricos Contínuos
# #### Analisando a feature (coluna) de PREÇO cobrado pelo host (price)
diagrama_caixa('price')
histograma('price')

# #### Removendo as linhas que contém outliers de preço
df_base_airbnb, linhas_removidas = remove_outliers('price')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
histograma('price')

# ### Analisando a feature (coluna) de PESSOAS ADICIONAIS (extra_people)
diagrama_caixa('extra_people')
histograma('extra_people')
df_base_airbnb, linhas_removidas = remove_outliers('extra_people')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
histograma('extra_people')


# ## Analise de Valores Numéricos Discretos
# ### Estas são as features que serão analisadas abaixo:
# 
# - host_listings_count -------------> float64 <br>
# - accommodates -------------> int64 <br>
# - bathrooms -------------> float64 <br>
# - bedrooms  -------------> float64 <br>
# - beds -------------> float64 <br>
# - guests_included -------------> int64 <br>
# - minimum_nights -------------> int64 <br>
# - maximum_nights -------------> int64 <br>
# - number_of_reviews -------------> int64 <br>
# - mes -------------> int64 <br>
# - ano -------------> int64 <br>


# ### Analisando a feature (coluna) de Quantidade de Imóveis Listados no Airbnb (host_listings_count)
diagrama_caixa('host_listings_count')
barras('host_listings_count')
df_base_airbnb, linhas_removidas = remove_outliers('host_listings_count')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
barras('host_listings_count')

# ### Analisando a feature (coluna) de Capacidade de Hospedes (accommodates)
diagrama_caixa('accommodates')
barras('accommodates')
df_base_airbnb, linhas_removidas = remove_outliers('accommodates')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
barras('accommodates')

# ### Analisando a feature (coluna) de Quantidade de Banheiros (bathrooms)
diagrama_caixa('bathrooms')
barras('bathrooms')
df_base_airbnb, linhas_removidas = remove_outliers('bathrooms')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
barras('bathrooms')

# ### Analisando a feature (coluna) de Quantidade de Quartos (bedrooms)
diagrama_caixa('bedrooms')
barras('bedrooms')
df_base_airbnb, linhas_removidas = remove_outliers('bedrooms')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
barras('bedrooms')


# ### Analisando a feature (coluna) de Quantidade de Camas (beds)
diagrama_caixa('beds')
barras('beds')
df_base_airbnb, linhas_removidas = remove_outliers('beds')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
barras('beds')

# ### Analisando a feature (coluna) de Quantidade de Pessoas Inclusas no Preço (guest_included)
diagrama_caixa('guests_included')
barras('guests_included')
df_base_airbnb.drop(columns='guests_included', axis=1, inplace=True)
print("Coluna 'guests_included' excluída do dataframe.")


df_base_airbnb.shape

# ### Analisando a feature (coluna) de Quantidade Minima de Diárias (minimum_nights)
diagrama_caixa('minimum_nights')
barras('minimum_nights')
df_base_airbnb, linhas_removidas = remove_outliers('minimum_nights')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
barras('minimum_nights')

# ### Analisando a feature (coluna) de Quantidade Máxima de Diárias (maximum_nights)
diagrama_caixa('maximum_nights')
barras('maximum_nights')
df_base_airbnb.drop(columns='maximum_nights', axis=1, inplace=True)
print("Coluna 'maximum_nights' excluída do dataframe.")
df_base_airbnb.shape

# ### Analisando a feature (coluna) de Número de Reviews (number_of_reviews)
diagrama_caixa('number_of_reviews')
barras('number_of_reviews')
df_base_airbnb.drop(columns='number_of_reviews', axis=1, inplace=True)
print("Coluna 'maximum_nights' excluída do dataframe.")
df_base_airbnb.shape

# ## Analise de Colunas com Valores em Texto

# Precisamos analisar os valores em texto e entender se vale a pena transformá-los em categorias para o nosso modelo. Para isso, vamos contar quantos tipos existem e quantas vezes aparecem no dataframe.

# ### Analisando a feature (coluna) de Tipo de Propriedade (property_type)

print(df_base_airbnb['property_type'].value_counts()) # soma a quantidade de ocorrências de cada tipo de propriedade

for tipo in df_base_airbnb['property_type'].value_counts().index: # percorre cada tipo de propriedade do dataframe 
    if df_base_airbnb['property_type'].value_counts()[tipo] < 2000:
        df_base_airbnb.loc[df_base_airbnb['property_type'] == tipo, 'property_type'] = 'Other' 
        # substitui, na coluna property_type o tipo de propriedade por 'Other' se a quantidade de ocorrências for menor que 2000
print(df_base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
oi = sns.countplot(x='property_type', data=df_base_airbnb, order=df_base_airbnb['property_type'].value_counts().index)
oi.tick_params(axis='x',rotation=90)

print(df_base_airbnb['room_type'].value_counts()) # soma a quantidade de ocorrências de cada tipo de quarto

# ### Analisando a feature (coluna) de Tipo de Cama (bed_type)
print(df_base_airbnb['bed_type'].value_counts()) # soma a quantidade de ocorrências de cada tipo de cama

for tipo in df_base_airbnb['bed_type'].value_counts().index: # percorre cada tipo de cama do dataframe 
    if df_base_airbnb['bed_type'].value_counts()[tipo] < 10000:
        df_base_airbnb.loc[df_base_airbnb['bed_type'] == tipo, 'bed_type'] = 'Other_Bed_Type' 
        # substitui, na coluna bed_type o tipo de propriedade por 'Other Bed Type' se a quantidade de ocorrências for menor que 10000
print(df_base_airbnb['bed_type'].value_counts())

# ### Analisando a feature (coluna) de Política de Cancelamento (cancellation_policy)
print(df_base_airbnb['cancellation_policy'].value_counts()) # soma a quantidade de ocorrências de cada política de cancelamento
for tipo in df_base_airbnb['cancellation_policy'].value_counts().index: # percorre cada tipo de cama do dataframe 
    if df_base_airbnb['cancellation_policy'].value_counts()[tipo] < 10000:
        df_base_airbnb.loc[df_base_airbnb['cancellation_policy'] == tipo, 'cancellation_policy'] = 'strict' 
        # substitui, na coluna cancellation_policy o tipo de politica de cancelamento por 'strict' se a quantidade de ocorrências for menor que 10000
print(df_base_airbnb['cancellation_policy'].value_counts())


# ### Analisando a feature (coluna) de Comodidades (ammenities)
df_base_airbnb['n_amenities'] = df_base_airbnb['amenities'].str.split(',').apply(lambda x: len(x)) 
# cria uma nova coluna 'n_ammenities' que conta a quantidade de amenidades em cada registro
df_base_airbnb.drop(columns='amenities', axis=1, inplace=True) # remove a coluna 'amenities' do dataframe
df_base_airbnb.shape

# Agora vamos analisar a nova coluna  na busca de outliers.
diagrama_caixa('n_amenities')
barras('n_amenities')
df_base_airbnb, linhas_removidas = remove_outliers('n_amenities')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
barras('n_amenities')

# ## Visualização de Mapa da Propriedades

amostra = df_base_airbnb.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_map(amostra, lat='latitude', lon='longitude',z='price', radius=5,
                        center=centro_mapa, zoom=10,
                        map_style='open-street-map',)
mapa.show()

# ### Encoding

df_base_airbnb.head()
print(df_base_airbnb.columns)
colunas_booleanas = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
df_base_airbnb_cod =  df_base_airbnb.copy() # Cria uma cópia do dataframe original para evitar alterações indesejadas

for coluna in colunas_booleanas:
    df_base_airbnb_cod[coluna] = df_base_airbnb_cod[coluna].map({'t': 1, 'f': 0})
    # Converte as colunas booleanas de 't' e 'f' para 1 e 0
colunas_categoricas = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
df_base_airbnb_cod = pd.get_dummies(df_base_airbnb_cod, columns=colunas_categoricas, dtype=float)
# Converte as colunas categóricas em variáveis dummy, removendo a primeira categoria para evitar a armadilha da variável fictícia

print(df_base_airbnb_cod.head())

# ### Modelo de Previsão
def avaliar_modelo(nome_modelo, y_teste, y_pred):
    """
    Avalia o desempenho do modelo com base no R² e no erro quadrático médio (RMSE).
    """
    r2 = r2_score(y_teste, y_pred)
    rmse = np.sqrt(mean_squared_error(y_teste, y_pred))
    
    return f'Modelo: {nome_modelo} --> R² = {r2} , RMSE = {rmse}'

modelo_rl = LinearRegression()
modelo_rf = RandomForestRegressor()
modelo_et = ExtraTreesRegressor()


modelos = {'Regressão Linear': modelo_rl,
           'Random Forest': modelo_rf,
           'Extra Trees': modelo_et}

x = df_base_airbnb_cod.drop(columns='price', axis=1)  # Variáveis independentes
y = df_base_airbnb_cod['price']  # Variável dependente


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=27) # testar com test_size=0.2 para 20% de teste

for nome_modelo, modelo in modelos.items(): # modelos.items() retorna uma lista de tuplas (nome_modelo, modelo)
    # Treino
    modelo.fit(x_train, y_train)  # Treina o modelo
    # Teste
    y_pred = modelo.predict(x_test)  # Faz previsões no conjunto de teste
    # Avaliação
    print(avaliar_modelo(nome_modelo, y_test, y_pred))  # Avalia o modelo


# ### Ajustes e Melhorias no Melhor Modelo

df_importancia_cada_coluna = pd.DataFrame(modelo_et.feature_importances_, index=x_train.columns) 
# Cria um DataFrame com as importâncias das colunas(features) do modelo Extra Trees Regressor, que foi avaliado o melhor.
df_importancia_cada_coluna = df_importancia_cada_coluna.sort_values(by=0, ascending=False)  
# Ordena as importâncias em ordem decrescente
print(df_importancia_cada_coluna)

plt.figure(figsize=(15, 5))
barras_features_importances = sns.barplot(x=df_importancia_cada_coluna.index, y=df_importancia_cada_coluna[0])
barras_features_importances.tick_params(axis='x', rotation=90)  # Rotaciona os rótulos do eixo x para melhor legibilidade
plt.title('Importância de cada coluna no modelo Extra Trees')


df_base_airbnb_cod = df_base_airbnb_cod.drop(columns=['is_business_travel_ready'], axis=1)

x = df_base_airbnb_cod.drop(columns='price', axis=1)  # Variáveis independentes
y = df_base_airbnb_cod['price']  # Variável dependente

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=27)

modelo_et.fit(x_train, y_train)  # Treina o modelo
    # Teste
y_pred = modelo_et.predict(x_test)  # Faz previsões no conjunto de teste
    # Avaliação
print(avaliar_modelo('Extra Trees', y_test, y_pred)) 


# #### Salvando a base de dados Tratada

x['price'] = y  # Adiciona a coluna 'price' ao DataFrame x para visualização
x.to_csv('base_de_dados_tratada.csv', index=False, sep=',', encoding='utf-8-sig')

# #### Salvando a base de dados de teste com as previsões feitas pelo modelo Extra Trees em sua última rodada.

x_aux = x_test.copy()
x_aux['price'] = y_test  # Adiciona a coluna 'price' com os valores reais do conjunto de teste
x_aux['previsoes_do_modelo_extra_trees'] = y_pred # Adiciona a coluna 'previsao' com as previsões do modelo
x_aux.to_csv('com_previsoes_teste_do_melhor_modelo_base_de_dados.csv', index=False, sep=',', encoding='utf-8-sig')

# #### Perpetuando o modelo em um arquivo

import joblib
joblib.dump(modelo_et, 'modelo_extra_trees_airbnb.joblib')