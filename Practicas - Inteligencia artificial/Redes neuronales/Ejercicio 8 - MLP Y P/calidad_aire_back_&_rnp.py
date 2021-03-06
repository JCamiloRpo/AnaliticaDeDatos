# -*- coding: utf-8 -*-
"""Calidad Aire - Back & RNP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j8DMRu43wttDNuB1vgs_1N4U6_0Uulg-
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
data = pd.read_excel("prediccionCalidadAire.xlsx", sheet_name=0) # sheet_name = 0 - primera hoja
data.head()

#Conocemos los datos
data.info() #Informacion sobre cada columna, tipo y numero de registros

#Corrección del tipo de datos
data['outlook']=data['outlook'].astype('category')
data['Alert']=data['Alert'].astype('category')
data.info()

#Descripción de variables numéricas
print(data.describe());
data.plot.hist(bins=5);

#Descripción variables categóricas
data['outlook'].value_counts().plot(kind='bar')

#Descripción variables categóricas
data['Alert'].value_counts().plot(kind='bar')

#Sklearn sólo analiza variables numéricas entonces se pasan las variables tipo category a numeros
dummiesOutlook = pd.get_dummies(data['outlook']) 
data = data.drop('outlook', axis=1)
data = data.join(dummiesOutlook)
data.head()

#Se codifican las categorias de la variable objetivo
data["Alert"]=data["Alert"].replace({"Yes": 1, "No": 0}) 
data.head()

"""# 2. División 70-30"""

#División 70-30
from sklearn.model_selection import train_test_split
X = data.drop("Alert", axis = 1)
Y = data['Alert']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
Y_train.value_counts()

"""# 3. Aprendizaje del Modelo: A)MLP"""

#Creación del modelo con el conjunto de entrenamiento
from sklearn.neural_network import MLPClassifier
modelMLP = MLPClassifier(activation = "identity", hidden_layer_sizes = (4), 
                     learning_rate = "adaptive", learning_rate_init = 0.01,
                     momentum = 0.09, max_iter = 200, verbose = True, random_state = 1)
modelMLP.fit(X_train, Y_train)

"""# B) Deep Learning"""

#Creación del modelo con el conjunto de entrenamiento
from keras.models import Sequential
from keras.layers.core import Dense
#Red neuronal
modelDL = Sequential()
modelDL.add(Dense(8, input_dim = 6, activation = 'sigmoid'))
modelDL.add(Dense(2, activation = 'sigmoid'))
modelDL.add(Dense(1, activation = 'relu')) # Capa de salida
modelDL.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
modelDL.fit(X_train, Y_train, epochs=100)

"""# 4. Evaluación del modelo sobre el conjunto de prueba
- Exactitud
"""

#Evaluación sobre el conjunto de prueba
from sklearn import metrics

Y_pred = modelMLP.predict(X_test) 

acc=metrics.accuracy_score(Y_test, Y_pred)
print(f'Exactitud: {acc}')

#Evaluación sobre el conjunto de prueba
from sklearn import metrics

print(f'Exactitud: {modelDL.evaluate(X_test,Y_test)}') #Exclusiva de Redes neuronales

"""# 5. Predicción para datos futuros

- Cargamos los datos futuros
- Aplicamos el modelo para la predicción
"""

#Cargamos los datos futuros
data_fut = pd.read_excel("prediccionCalidadAire-futuro.xlsx", sheet_name=0)
data_fut.head()

dummiesOutlookFut = pd.get_dummies(data_fut['outlook']) 
data_fut = data_fut.drop('outlook', axis=1)
data_fut = data_fut.join(dummiesOutlookFut)
data_fut.head()

#Hacemos la predicción
Y_fut = modelMLP.predict(data_fut)
print(Y_fut)

#Hacemos la predicción
Y_fut = modelDL.predict(data_fut)
print(Y_fut.round())