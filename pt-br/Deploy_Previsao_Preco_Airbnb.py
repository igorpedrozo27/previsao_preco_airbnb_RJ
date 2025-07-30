import pandas as pd
import streamlit as st
import joblib

# criação de dicionários com as features necessárias para a previsão para que o usuário possa preencher os dados do imóvel

x_numericos = {'latitude':0, 'longitude':0, 'accommodates':0, 'bathrooms':0, 'bedrooms':0,'beds':0,
               'extra_people':0, 'minimum_nights':0,'ano':0, 'mes':0,'n_amenities':0, 'host_listings_count': 0}

x_booleanos = {'host_is_superhost':0, 'instant_bookable':0}

x_categoricos = {'property_type':['Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite',
                                  'Guesthouse', 'Hostel', 'House', 'Loft', 'Other', 'Serviced apartment'],
                'room_type':['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
                'bed_type':['Other_Bed_Type', 'Real Bed'],
                'cancellation_policy':['flexible', 'moderate', 'strict', 'strict_14_with_grace_period']
                }

dic_aux_var_dumm = {} # dicionário auxiliar para armazenar as variáveis dummy que serão criadas a partir das variáveis categóricas

for item in x_categoricos:
    for opcao in x_categoricos[item]:
        dic_aux_var_dumm[f'{item}_{opcao}'] = 0 # cria uma coluna no dicionario auxiliar para cada opção de variável categórica, inicializando com 0

for item in x_numericos: # ajusta o input do usuario para valores numericos
    if item == 'latitude' or item == 'longitude':
        x_numericos[item] = st.number_input(f'{item}', step=0.000001, value=float(x_numericos[item]), format="%.6f")
    elif item == 'extra_people':
        x_numericos[item] = st.number_input(f'{item}', step=0.01, value=float(x_numericos[item]), format="%.2f")
    else:
        x_numericos[item] = st.number_input(f'{item}', value=x_numericos[item])

for item in x_booleanos: # ajusta o input do usuario para valores booleanos
    x_booleanos[item] = st.selectbox(f'{item}', options=['Sim', 'Não'], index=0)
    if x_booleanos[item] == 'Sim':
        x_booleanos[item] = 1
    else:
        x_booleanos[item] = 0

for item in x_categoricos:
    x_categoricos[item] = st.selectbox(f'{item}', options=x_categoricos[item], index=0)
    dic_aux_var_dumm[f'{item}_{opcao}'] = 1 # atualiza o dicionário auxiliar com a opção selecionada pelo usuário

botao_previsao = st.button('Prever Preço do Imóvel') # cria o botao para fazer a previsão do preço do imóvel

if botao_previsao: # se houver clique no botao de previsao, executa o seguinte bloco de código
    dic_aux_var_dumm.update(x_numericos) # une o dicionário de variáveis categoricas com o de variaveis numericas
    dic_aux_var_dumm.update(x_booleanos) # une o dicionario atualizado na linha acima com o dicionario de variáveis booleanas
    df_valores_preenchidos_usuario = pd.DataFrame([dic_aux_var_dumm]) 
    # cria um dataframe dos dicionarios acima, que contem as informacoes preenchidas pelo usuário que serao usadas para fazer a previsao
    dados = pd.read_csv('base_de_dados_tratada.csv')
    colunas = list(dados.columns)[:-1]  # Exclui 'price'
    df_valores_preenchidos_usuario = df_valores_preenchidos_usuario[colunas]
    modelo = joblib.load('modelo_extra_trees_airbnb.joblib')
    previsao = modelo.predict(df_valores_preenchidos_usuario)
    st.write(f'Preço previsto segundo este modelo para o imóvel com as características descritas é de: R$ {previsao[0]:.2f}')