import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from imblearn.over_sampling import RandomOverSampler

# Clase para la transformación de fecha y codificación categórica
class DateEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['Date'] = pd.to_datetime(X['Date'])
        X['Day'] = X['Date'].dt.day
        X['Month'] = X['Date'].dt.month
        X['Year'] = X['Date'].dt.year
        X.drop(columns=['Date'], inplace=True)
        return X

# Clase para la codificación de variables categóricas
class GetDummiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_encode = None
        self.dummy_columns = None
    
    def fit(self, X, y=None):
        self.columns_to_encode = X.select_dtypes(include=['object', 'bool']).columns
        self.dummy_columns = pd.get_dummies(X[self.columns_to_encode], drop_first=True).columns
        return self
    
    def transform(self, X):
        dummies = pd.get_dummies(X[self.columns_to_encode], drop_first=True)
        for col in self.dummy_columns:
            if col not in dummies.columns:
                dummies[col] = 0
        dummies = dummies[self.dummy_columns]
        non_categorical_cols = X.drop(self.columns_to_encode, axis=1)
        transformed_data = pd.concat([non_categorical_cols, dummies], axis=1)
        return transformed_data

# Clase para el escalado de características
class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        scaled_data = self.scaler.transform(X)
        return pd.DataFrame(scaled_data)

# Clase para construir el modelo de regresión neuronal
class NeuralNetworkRegression(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    def fit(self, X, y, epochs=150, batch_size=64):
        self.model = Sequential([
            Input(shape=(X.shape[1],)),
            Dense(14, activation='relu'),
            Dropout(0.167),
            Dense(17, activation='relu'),
            Dropout(0.167),
            Dense(1, activation='linear')
        ])

        self.model.add(BatchNormalization())
        
        optimizer = Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_model(self):
        return self.model
    
# Clase para construir el modelo de clasificación con Random Forest
class NeuralNetworkClasification(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y, epochs=70, batch_size=16):
        self.model = Sequential([
            Input(shape=(X.shape[1],)),
            Dense(14, activation='sigmoid'),
            Dropout(0.278),
            Dense(7, activation='sigmoid'),
            Dropout(0.278),
            Dense(1, activation='sigmoid'),
        ])

        self.model.add(BatchNormalization())
        
        optimizer = Adam(learning_rate=0.006)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['F1Score'])
        
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
    
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)
    
    def get_model(self):
        return self.model