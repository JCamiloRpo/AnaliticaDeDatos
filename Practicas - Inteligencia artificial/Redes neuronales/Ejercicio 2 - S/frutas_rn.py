# -*- coding: utf-8 -*-
"""Frutas - RN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XITXqCgheGNH3hpE0xv_6kQQBsYb-kEx

# Aprendizaje Supervisado - Redes Neuronales

1. Preparación de Datos
2. División de los datos
3. Aprendizaje del Modelo
4. Evaluación del Modelo
5. Predicción de Datos Futuros
"""

#Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""# 1. Preparación de Datos
-  Cargamos los datos
-  Conocemos los datos con estadísticos
- Transformación de datos  (sklearn sólo analiza variables numéricas)
"""

#Cargamos los datos
data = pd.read_excel("frutas.xlsx", sheet_name="frutas")
data.head()

#Conocemos los datos
data.info()

#Corrección del tipo de datos
data['y']=data['y'].astype('category') #Se corrige el tipo de dato object a category dataframe['Columna']
data.info()

#Descripción de variables numéricas
print(data.describe());
data.plot.hist(bins=5);

#Descripción variables categóricas
data['y'].value_counts().plot(kind='bar')

"""# 2. División 70-30"""

#División 70-30
from sklearn.model_selection import train_test_split
X = data.drop("y", axis = 1) 
Y = data['y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
Y_train.value_counts()

"""# 3. Aprendizaje del Modelo: Red Neuronal"""

#Creación del modelo con el conjunto de entrenamiento
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(activation = "identity", hidden_layer_sizes = (1), 
                     learning_rate = "adaptive", learning_rate_init = 0.01,
                     momentum = 0.09, max_iter = 200, verbose = True, random_state = 1)
model.fit(X_train, Y_train)

print(model.coefs_)
print(model.intercepts_)

"""# 4. Evaluación del modelo sobre el conjunto de prueba
- Exactitud
"""

#Evaluación sobre el conjunto de prueba
from sklearn import metrics

Y_pred = model.predict(X_test)

acc=metrics.accuracy_score(Y_test, Y_pred)
print(f'Exactitud: {acc}')

"""# 5. Predicción para datos futuros

- Cargamos los datos futuros
- Aplicamos el modelo para la predicción
"""

#Cargamos los datos futuros
data_fut = pd.read_excel("frutas.xlsx", sheet_name = "datos futuros")
data_fut.head()

#Hacemos la predicción
Y_fut = model.predict(data_fut)
print(Y_fut)