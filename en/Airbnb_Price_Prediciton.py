import pandas as pd # library for data analysis and manipulation
import pathlib as pl # library for traversing files on the computer
import numpy as np # library for arrays and mathematical operations
import seaborn as sns # library for data visualization
import matplotlib.pyplot as plt # library for data visualization
import plotly.express as px # library for data visualization

# machine learning libraries, prediction models and model performance evaluators
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# Create a dictionary to convert months to numerical format
# We'll use the first 3 characters of the filename, which contain the abbreviated months
# Later, we'll use this to convert the abbreviated month names to numbers

meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
         'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

# Define the path to CSV files

caminho = pl.Path(r'C:/Users/igor_/Programacao/Cursos Hashtag/Python/Projeto de Ciencia de Dados e Recomendacoes/dataset/dataset/')

# Create an empty dataframe to store data from all CSV files

df_base_airbnb = pd.DataFrame()

# Iterate over files in the specified directory, performing the following operations for each file:

for arquivo in caminho.iterdir():

    df_aux = pd.read_csv(caminho / arquivo) # create an auxiliary dataframe for each CSV file
    df_aux['mes'] = meses[arquivo.name[:3]] # Add a 'month' column with the corresponding month number from the filename
    df_aux['ano'] = int(arquivo.name[-8:].replace('.csv','')) # Add a 'year' column with the year extracted from the filename

    df_base_airbnb = pd.concat([df_base_airbnb, df_aux], ignore_index=True) # Concatena o dataframe auxiliar ao dataframe principal, ignorando os índices
        
# Exibe as primeiras linhas do dataframe resultante
df_base_airbnb.head()

# ### Consolidate Database for Qualitative Analysis
# A sample of the complete database will be created and exported to CSV for analysis and removal of unnecessary features (columns) for the project's final objective in Excel.

df_base_airbnb.head(1000).to_csv('base_airbnb_primeiros_registros.csv', index=False, sep=',', encoding='utf-8-sig')

# Columns where all or almost all values are equal - 'host_listings_count' and 'host_total_listings_count'

# Compare 'host_listings_count' with 'host_total_listings_count' and print the result
print((df_base_airbnb['host_listings_count']==df_base_airbnb['host_total_listings_count']).value_counts())

# Check and print how many entries in the 'square_feet' column are null

print(df_base_airbnb['square_feet'].isnull().sum())

# After analysis in Excel with the exported file,
# 
# - Types of columns excluded:
#     1. IDs, Links, and information not relevant to the model
#     2. Columns that provide similar information (Ex: Date vs Year/Month)

# ### Filtering the Database after Qualitative Analysis - column exclusion done directly in Excel.

colunas_remanescentes = 'host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','mes','ano'

# Select only the remaining columns from the df_base_airbnb dataframe

df_base_airbnb = df_base_airbnb.loc[:, colunas_remanescentes]
print(df_base_airbnb)

# ### Handling Missing Values
print(df_base_airbnb.isnull().sum())

# iterate through the database and exclude columns that have more than 100,000 null values
for coluna in df_base_airbnb:
    if (df_base_airbnb[coluna].isnull().sum()) > 100000:
        df_base_airbnb.drop(columns=coluna, axis=1, inplace=True)
        print(f'Column {coluna} excluded')

print('=============================================')
print(df_base_airbnb.isnull().sum())
# ### Removing Rows with Null Values

print(df_base_airbnb.shape)

linhas_ant = df_base_airbnb.shape[0] # store the initial number of rows from the shape tuple

df_base_airbnb = df_base_airbnb.dropna() # remove rows that have null values in any column of the dataframe

linhas_post = df_base_airbnb.shape[0] # store the final number of rows from the shape tuple, 

print(f'Novo shape do dataframe (linhas, colunas): {df_base_airbnb.shape}')

print(f'{linhas_ant - linhas_post} linhas foram removidas do dataframe')

# NOTE: dropna() removes all rows that have at least one null value. If you want to remove only rows that have null values in all columns, use the parameter how='all' in the dropna() method

print(df_base_airbnb.isnull().sum())
# ### Checking Data Types in Each Column
print(df_base_airbnb.info())

print(df_base_airbnb.dtypes)
print('='*100)

print(df_base_airbnb.loc[0,:])
# ### Changing Data Types of Variables from Object to Float
# Remove '$' and ',' and convert to float
df_base_airbnb['price'] = df_base_airbnb['price'].str.replace('[$,]', '', regex=True).astype(np.float32, copy=False)

df_base_airbnb['extra_people'] = df_base_airbnb['extra_people'].str.replace('[$,]', '', regex=True).astype(np.float32, copy=False)

print(df_base_airbnb.dtypes)

# ### Exploratory Analysis and Outlier Treatment
# ##### Plotting the Feature Correlation Matrix

plt.figure(figsize=(15,5))
sns.heatmap(df_base_airbnb.corr(numeric_only=True), annot=True, cmap='coolwarm')

# ##### - For outlier treatment, we will apply the <b>Interquartile Range</b> method of Descriptive Statistics

# The Interquartile Range (IQR) method is a widely used descriptive statistics technique for:
#
# - Summarizing data dispersion (variability)
# - Identifying outliers (discrepant values)
#
# The IQR is the difference between the third quartile (Q3) and the first quartile (Q1) of an ordered dataset:
#
# - Interquartile Range:
#     - <b>IQR = Q3 − Q1</b>
#
#         - Q1 (1st quartile): value below which 25% of the data lies
#         - Q3 (3rd quartile): value below which 75% of the data lies
#         - IQR: represents the "middle range" of the central 50% of the sample
#
# Values are considered outliers if they are very distant from the center of the data.
#
# - Lower limit:
#     - <b>LB = Q1 − 1.5 × IQR</b>
# - Upper limit:
#     - <b>UB = Q3 + 1.5 × IQR</b>
#
# Any value outside these limits is an outlier.

# ### Defining functions to calculate lower and upper limits for analyzing each feature (column)

def limites (coluna):
    """
    Calculates the lower and upper limits for the specified column.
    Uses the IQR (Interquartile Range) method to detect outliers.
    """
    Q1 = df_base_airbnb[coluna].quantile(0.25)
    Q3 = df_base_airbnb[coluna].quantile(0.75)
    IQR = Q3 - Q1 # interquartile range
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
    Plots a box plot for the specified column.
    """  

def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.histplot(df_base_airbnb[coluna], kde=True, bins=50)
    """
    Plots a histogram for the specified column.
    """

def barras(coluna):
    plt.figure(figsize=(15, 5))
    sns.barplot(x=df_base_airbnb[coluna].value_counts().index, y=df_base_airbnb[coluna].value_counts())
    """
    Plots a bar chart for the specified column.
    """
# ## Analysis of Continuous Numerical Values
# #### Analyzing the Feature (Column) of PRICE charged by host (price)
diagrama_caixa('price')
histograma('price')

# #### Removing rows containing price outliers
df_base_airbnb, linhas_removidas = remove_outliers('price')
print(f"{linhas_removidas} rows were removed from the dataframe due to outliers in the feature (column).")
histograma('price')

# ### Analyzing the Feature (Column) of EXTRA PEOPLE (extra_people)
diagrama_caixa('extra_people')
histograma('extra_people')
df_base_airbnb, linhas_removidas = remove_outliers('extra_people')
print(f"{linhas_removidas} linhas foram removidas do dataframe devido a outliers da feature(coluna).")
histograma('extra_people')


# ## Analysis of Discrete Numerical Values
# ### These are the features that will be analyzed below:
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
# - month -------------> int64 <br>
# - year -------------> int64 <br>


# ### Analyzing the Feature (Column) of Number of Properties Listed on Airbnb (host_listings_count)
diagrama_caixa('host_listings_count')
barras('host_listings_count')
df_base_airbnb, linhas_removidas = remove_outliers('host_listings_count')
print(f"{linhas_removidas} rows were removed from the dataframe due to outliers in the feature (column).")
barras('host_listings_count')

# ### Analyzing the Feature (Column) of Guest Capacity (accommodates)
diagrama_caixa('accommodates')
barras('accommodates')
df_base_airbnb, linhas_removidas = remove_outliers('accommodates')
print(f"{linhas_removidas} rows were removed from the dataframe due to outliers in the feature (column).")
barras('accommodates')

# ### Analyzing the Feature (Column) of Number of Bathrooms (bathrooms)
diagrama_caixa('bathrooms')
barras('bathrooms')
df_base_airbnb, linhas_removidas = remove_outliers('bathrooms')
print(f"{linhas_removidas} rows were removed from the dataframe due to outliers in the feature (column).")
barras('bathrooms')

# ### Analyzing the Feature (Column) of Number of Bedrooms (bedrooms)
diagrama_caixa('bedrooms')
barras('bedrooms')
df_base_airbnb, linhas_removidas = remove_outliers('bedrooms')
print(f"{linhas_removidas} rows were removed from the dataframe due to outliers in the feature (column).")
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

# ### Analyzing the Feature (Column) of Minimum Number of Nights (minimum_nights)
diagrama_caixa('minimum_nights')
barras('minimum_nights')
df_base_airbnb, linhas_removidas = remove_outliers('minimum_nights')
print(f"{linhas_removidas} rows were removed from the dataframe due to outliers in the feature (column).")
barras('minimum_nights')

# ### Analyzing the Feature (Column) of Maximum Number of Nights (maximum_nights)
diagrama_caixa('maximum_nights')
barras('maximum_nights')
df_base_airbnb.drop(columns='maximum_nights', axis=1, inplace=True)
print("Column 'maximum_nights' removed from the dataframe.")
df_base_airbnb.shape

# ### Analyzing the Feature (Column) of Number of Reviews (number_of_reviews)
diagrama_caixa('number_of_reviews')
barras('number_of_reviews')
df_base_airbnb.drop(columns='number_of_reviews', axis=1, inplace=True)
print("Column 'number_of_reviews' removed from the dataframe.")
df_base_airbnb.shape

# ## Analysis of Text Value Columns

# We need to analyze the text values and understand if it's worth transforming them into categories for our model. For this, we'll count how many types exist and how many times they appear in the dataframe.

# ### Analyzing the Feature (Column) of Property Type (property_type)

print(df_base_airbnb['property_type'].value_counts()) # counts the number of occurrences of each property type

for tipo in df_base_airbnb['property_type'].value_counts().index: # iterates through each property type in the dataframe
    if df_base_airbnb['property_type'].value_counts()[tipo] < 2000:
        df_base_airbnb.loc[df_base_airbnb['property_type'] == tipo, 'property_type'] = 'Other' 
        # replaces property types with 'Other' in the property_type column if they appear less than 2000 times
print(df_base_airbnb['property_type'].value_counts())

plt.figure(figsize=(15, 5))
oi = sns.countplot(x='property_type', data=df_base_airbnb, order=df_base_airbnb['property_type'].value_counts().index)
oi.tick_params(axis='x',rotation=90)

print(df_base_airbnb['room_type'].value_counts()) # counts the number of occurrences of each room type

# ### Analyzing the Feature (Column) of Bed Type (bed_type)
print(df_base_airbnb['bed_type'].value_counts()) # counts the number of occurrences of each bed type

for tipo in df_base_airbnb['bed_type'].value_counts().index: # iterates through each bed type in the dataframe
    if df_base_airbnb['bed_type'].value_counts()[tipo] < 10000:
        df_base_airbnb.loc[df_base_airbnb['bed_type'] == tipo, 'bed_type'] = 'Other_Bed_Type' 
        # replaces bed types with 'Other_Bed_Type' in the bed_type column if they appear less than 10000 times
print(df_base_airbnb['bed_type'].value_counts())

# ### Analyzing the Feature (Column) of Cancellation Policy (cancellation_policy)
print(df_base_airbnb['cancellation_policy'].value_counts()) # counts the number of occurrences of each cancellation policy
for tipo in df_base_airbnb['cancellation_policy'].value_counts().index: # iterates through each cancellation policy type in the dataframe
    if df_base_airbnb['cancellation_policy'].value_counts()[tipo] < 10000:
        df_base_airbnb.loc[df_base_airbnb['cancellation_policy'] == tipo, 'cancellation_policy'] = 'strict' 
        # replaces cancellation policy types with 'strict' in the cancellation_policy column if they appear less than 10000 times
print(df_base_airbnb['cancellation_policy'].value_counts())


# ### Analyzing the Feature (Column) of Amenities (amenities)
df_base_airbnb['n_amenities'] = df_base_airbnb['amenities'].str.split(',').apply(lambda x: len(x)) 
# creates a new column 'n_amenities' that counts the number of amenities in each record
df_base_airbnb.drop(columns='amenities', axis=1, inplace=True) # removes the 'amenities' column from the dataframe
df_base_airbnb.shape

# Now let's analyze the new column for outliers.
diagrama_caixa('n_amenities')
barras('n_amenities')
df_base_airbnb, linhas_removidas = remove_outliers('n_amenities')
print(f"{linhas_removidas} rows were removed from the dataframe due to outliers in the feature (column).")
barras('n_amenities')

# ## Properties Map Visualization

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
df_base_airbnb_cod =  df_base_airbnb.copy() # Creates a copy of the original dataframe to avoid unwanted changes

for coluna in colunas_booleanas:
    df_base_airbnb_cod[coluna] = df_base_airbnb_cod[coluna].map({'t': 1, 'f': 0})
    # Converts boolean columns from 't' and 'f' to 1 and 0
colunas_categoricas = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
df_base_airbnb_cod = pd.get_dummies(df_base_airbnb_cod, columns=colunas_categoricas, dtype=float)
# Converts categorical columns into dummy variables, removing the first category to avoid the dummy variable trap

print(df_base_airbnb_cod.head())

# ### Prediction Model
def avaliar_modelo(nome_modelo, y_teste, y_pred):
    """
    Evaluates model performance based on R² and Root Mean Square Error (RMSE).
    """
    r2 = r2_score(y_teste, y_pred)
    rmse = np.sqrt(mean_squared_error(y_teste, y_pred))
    
    return f'Model: {nome_modelo} --> R² = {r2} , RMSE = {rmse}'

modelo_rl = LinearRegression()
modelo_rf = RandomForestRegressor()
modelo_et = ExtraTreesRegressor()


modelos = {'Linear Regression': modelo_rl,
           'Random Forest': modelo_rf,
           'Extra Trees': modelo_et}

x = df_base_airbnb_cod.drop(columns='price', axis=1)  # Independent variables
y = df_base_airbnb_cod['price']  # Dependent variable


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=27) # test with test_size=0.2 for 20% test data

for nome_modelo, modelo in modelos.items(): # modelos.items() returns a list of tuples (model_name, model)
    # Training
    modelo.fit(x_train, y_train)  # Train the model
    # Testing
    y_pred = modelo.predict(x_test)  # Make predictions on the test set
    # Evaluation
    print(avaliar_modelo(nome_modelo, y_test, y_pred))  # Evaluate the model


# ### Adjustments and Improvements to the Best Model

df_importancia_cada_coluna = pd.DataFrame(modelo_et.feature_importances_, index=x_train.columns) 
# Creates a DataFrame with the feature importances from the Extra Trees Regressor model, which was evaluated as the best.
df_importancia_cada_coluna = df_importancia_cada_coluna.sort_values(by=0, ascending=False)  
# Sorts the importances in descending order
print(df_importancia_cada_coluna)

plt.figure(figsize=(15, 5))
barras_features_importances = sns.barplot(x=df_importancia_cada_coluna.index, y=df_importancia_cada_coluna[0])
barras_features_importances.tick_params(axis='x', rotation=90)  # Rotates x-axis labels for better readability
plt.title('Feature Importance in the Extra Trees Model')


df_base_airbnb_cod = df_base_airbnb_cod.drop(columns=['is_business_travel_ready'], axis=1)

x = df_base_airbnb_cod.drop(columns='price', axis=1)  # Independent variables
y = df_base_airbnb_cod['price']  # Dependent variable

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=27)

modelo_et.fit(x_train, y_train)  # Train the model
    # Test
y_pred = modelo_et.predict(x_test)  # Make predictions on the test set
    # Evaluation
print(avaliar_modelo('Extra Trees', y_test, y_pred)) 


# #### Saving the Processed Database

x['price'] = y  # Adds the 'price' column to DataFrame x for visualization
x.to_csv('processed_database.csv', index=False, sep=',', encoding='utf-8-sig')

# #### Saving the Test Database with Predictions Made by the Extra Trees Model in its Last Run

x_aux = x_test.copy()
x_aux['price'] = y_test  # Adds the 'price' column with the actual test set values
x_aux['extra_trees_model_predictions'] = y_pred # Adds the 'prediction' column with the model's predictions
x_aux.to_csv('best_model_test_predictions_database.csv', index=False, sep=',', encoding='utf-8-sig')

# #### Persisting the Model to a File

import joblib
joblib.dump(modelo_et, 'modelo_extra_trees_airbnb.joblib')