import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from funciones_prediccion import DateEncoder, GetDummiesTransformer, FeatureScaler, NeuralNetworkRegression, NeuralNetworkClasification
from imblearn.over_sampling import RandomOverSampler

# Cargar datos
rain_data = pd.read_csv("DATASET/weatherAUS.csv")

# Filtrar ciudades
ciudades = ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne',
            'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport']
rain_data = rain_data[rain_data['Location'].isin(ciudades)]
rain_data.drop(rain_data.columns[0], axis=1, inplace=True)

# Mapea en 0 y 1 l variable objetivo para clasificación
rain_data.RainTomorrow = rain_data.RainTomorrow.map({'Yes':1, 'No':0})

# División x e y para regresión
x_train_r = rain_data[rain_data['Date'] < '2016-01-01'].copy()
y_train_r = x_train_r['RainfallTomorrow'].copy()
x_train_r.drop(['RainfallTomorrow','RainTomorrow'], axis=1, inplace=True)

# División x e y para clasificación
x_train_c = rain_data[rain_data['Date'] < '2016-01-01'].copy()
y_train_c = x_train_c['RainTomorrow'].copy()
x_train_c.drop(['RainfallTomorrow','RainTomorrow'], axis=1, inplace=True)

# Preprocesamiento de los datos
def fill_nan(x):
    # Identificar columnas categóricas y numéricas
    categ_var = x.select_dtypes(include=['object']).columns.tolist()  
    nume_var = x.select_dtypes(exclude=['object']).columns.tolist()  

    # Imputar valores faltantes en variables categóricas con la moda
    for var in categ_var:
        x[var] = x[var].fillna(x[var].mode()[0])

    # Imputar valores faltantes en variables numéricas con la mediana
    for column in nume_var:
        x[column] = x[column].fillna(x[column].median())

    return x

# Rellenar valores faltantes x e y para regresión
fill_nan(x_train_r)
mediana_r = np.nanmedian(y_train_r)
y_train_r = np.nan_to_num(y_train_r, nan=mediana_r)

# Rellenar valores faltantes x e y para clasificación
fill_nan(x_train_c)
moda_c = y_train_c.mode()
y_train_c = np.nan_to_num(y_train_c, nan=moda_c)

# Oversampling con RandomOverSampler
ros = RandomOverSampler(random_state=42)
x_train_c, y_train_c = ros.fit_resample(x_train_c, y_train_c)

# Definir el pipeline completo
pipeline = Pipeline([
    ('date_encoder', DateEncoder()),
    ('cat_encoder', GetDummiesTransformer()),
    ('scaler', FeatureScaler()),
    ('model', None)
])

# Modelos disponibles
neural_network_rmodel = NeuralNetworkRegression()
neural_network_cmodel = NeuralNetworkClasification()

# Ajustar el pipeline de regresión
reg_pipeline = pipeline.set_params(model=neural_network_rmodel)
train_regression_pipeline = reg_pipeline.fit(x_train_r, y_train_r)

# Ajustar el pipeline de clasificación
clf_pipeline = pipeline.set_params(model=neural_network_cmodel)
train_classification_pipeline = clf_pipeline.fit(x_train_c, y_train_c)

# Crear pipelines de predicción sin el escalador
predict_regression_pipeline = Pipeline([
    ('date_encoder', train_regression_pipeline.named_steps['date_encoder']),
    ('cat_encoder', train_regression_pipeline.named_steps['cat_encoder']),
    ('model', train_regression_pipeline.named_steps['model'])
])

predict_classification_pipeline = Pipeline([
    ('date_encoder', train_classification_pipeline.named_steps['date_encoder']),
    ('cat_encoder', train_classification_pipeline.named_steps['cat_encoder']),
    ('model', train_classification_pipeline.named_steps['model'])
])

# Guardar el pipeline y el modelo entrenado para regresión
joblib.dump(train_regression_pipeline, 'regression_train_pipeline.joblib')
joblib.dump(predict_regression_pipeline, 'regression_pred_pipeline.joblib')

# Guardar el pipeline y el modelo entrenado para clasificación
joblib.dump(train_classification_pipeline, 'classification_train_pipeline.joblib')
joblib.dump(predict_classification_pipeline, 'classification_pred_pipeline.joblib')