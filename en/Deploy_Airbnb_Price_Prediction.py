import pandas as pd
import streamlit as st
import joblib

# creation of dictionaries with the necessary features for prediction so that users can fill in property data

x_numericos = {'latitude':0, 'longitude':0, 'accommodates':0, 'bathrooms':0, 'bedrooms':0,'beds':0,
               'extra_people':0, 'minimum_nights':0,'ano':0, 'mes':0,'n_amenities':0, 'host_listings_count': 0}

x_booleanos = {'host_is_superhost':0, 'instant_bookable':0}

x_categoricos = {'property_type':['Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite',
                                  'Guesthouse', 'Hostel', 'House', 'Loft', 'Other', 'Serviced apartment'],
                'room_type':['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
                'bed_type':['Other_Bed_Type', 'Real Bed'],
                'cancellation_policy':['flexible', 'moderate', 'strict', 'strict_14_with_grace_period']
                }

dic_aux_var_dumm = {} # auxiliary dictionary to store dummy variables that will be created from categorical variables

for item in x_categoricos:
    for opcao in x_categoricos[item]:
        dic_aux_var_dumm[f'{item}_{opcao}'] = 0 # creates a column in the auxiliary dictionary for each categorical variable option, initializing with 0

for item in x_numericos: # adjusts user input for numeric values
    if item == 'latitude' or item == 'longitude':
        x_numericos[item] = st.number_input(f'{item}', step=0.000001, value=float(x_numericos[item]), format="%.6f")
    elif item == 'extra_people':
        x_numericos[item] = st.number_input(f'{item}', step=0.01, value=float(x_numericos[item]), format="%.2f")
    else:
        x_numericos[item] = st.number_input(f'{item}', value=x_numericos[item])

for item in x_booleanos: # adjusts user input for boolean values
    x_booleanos[item] = st.selectbox(f'{item}', options=['Sim', 'Não'], index=0)
    if x_booleanos[item] == 'Sim':
        x_booleanos[item] = 1
    else:
        x_booleanos[item] = 0

for item in x_categoricos:
    x_categoricos[item] = st.selectbox(f'{item}', options=x_categoricos[item], index=0)
    dic_aux_var_dumm[f'{item}_{opcao}'] = 1 # atualiza o dicionário auxiliar com a opção selecionada pelo usuário

botao_previsao = st.button('Predict Property Price') # creates the button to predict the property price

if botao_previsao: # if the prediction button is clicked, executes the following code block
    dic_aux_var_dumm.update(x_numericos) # merges the categorical variables dictionary with the numeric variables dictionary
    dic_aux_var_dumm.update(x_booleanos) # merges the updated dictionary from the line above with the boolean variables dictionary
    df_valores_preenchidos_usuario = pd.DataFrame([dic_aux_var_dumm]) 
    # creates a dataframe from the above dictionaries, containing the information filled by the user that will be used for prediction
    dados = pd.read_csv('base_de_dados_tratada.csv')
    colunas = list(dados.columns)[:-1]  # Excludes 'price'
    df_valores_preenchidos_usuario = df_valores_preenchidos_usuario[colunas]
    modelo = joblib.load('modelo_extra_trees_airbnb.joblib')
    previsao = modelo.predict(df_valores_preenchidos_usuario)
    st.write(f'Predicted price according to this model for the property with the described characteristics is: R$ {previsao[0]:.2f}')