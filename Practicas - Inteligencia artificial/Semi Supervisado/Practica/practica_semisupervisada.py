# -*- coding: utf-8 -*-
"""Practica SemiSupervisada.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tGvbC_4zHLH8XHXQIfhXMn5Sq0SKKwHd

# Aprendizaje Semi-Supervisado

1. Preparación de Datos(Datos etiquetados y Datos NO etiquetados)
2. División de los datos
3. Aprendizaje del Modelo(70% etiquetados+ Datos No Etiquetados)
4. Evaluación del Modelo
5. Predicción de Datos Futuros
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""# 1. Preparación de Datos
-  Cargamos los datos
-  Conocemos los datos con estadísticos
- Seleccion de variables
- Limpieza de atipicos
- Limpieza de nulos
- Transformación de datos  (sklearn sólo analiza variables numéricas)

#1A.Datos Etiquetados
"""

data_etiquetada = pd.read_excel("carRisk-semisupervisado.xlsx", sheet_name=0)
data_etiquetada.info()

#Corrección del tipo de datos
data_etiquetada['cartype']=data_etiquetada['cartype'].astype('category')
data_etiquetada['risk']=data_etiquetada['risk'].astype('category')
data_etiquetada.info()

#Descripción de variables numéricas
print(data_etiquetada.describe())
data_etiquetada.plot.hist(bins=5)

#Descripción variables categóricas
data_etiquetada['cartype'].value_counts().plot(kind='bar')#.valuer_Counts() conteo de registros .plot(kind='bar') grafico de barras

#Descripción variables categóricas
data_etiquetada['risk'].value_counts().plot(kind='bar')#.valuer_Counts() conteo de registros .plot(kind='bar') grafico de barras

#Sklearn sólo analiza variables numéricas
dummies = pd.get_dummies(data_etiquetada['cartype'])
data_etiquetada = data_etiquetada.drop('cartype', axis=1)
data_etiquetada = data_etiquetada.join(dummies)
data_etiquetada.head()

#Se codifican las categorias de la variable objetivo
data_etiquetada['risk']=data_etiquetada['risk'].replace({"high": 1, "low": 0})
data_etiquetada.head()

"""#1B.Datos no etiquetados"""

data = pd.read_excel("carRisk-semisupervisado.xlsx", sheet_name=1)
data.info()

#Corrección del tipo de datos
data['cartype']=data['cartype'].astype('category')
data.info()

#Descripción de variables numéricas
print(data.describe())
data.plot.hist(bins=5)

#Descripción variables categóricas
data['cartype'].value_counts().plot(kind='bar')#.valuer_Counts() conteo de registros .plot(kind='bar') grafico de barras

#Sklearn sólo analiza variables numéricas
dummies = pd.get_dummies(data['cartype'])
data = data.drop('cartype', axis=1)
data = data.join(dummies)
data.head()

"""# 2. División 70-30 de los Datos etiquetados"""

#División 70-30
from sklearn.model_selection import train_test_split
X = data_etiquetada.drop("risk", axis = 1) 
Y = data_etiquetada['risk']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)

conjunto_entrenamiento=pd.concat([X_train,Y_train],axis=1)
conjunto_entrenamiento.head()

"""# 3. Aprendizaje del Modelo: Label Propagation"""

#Unimos los datos (70% etiquetados+No etiquetados)
data_Total=pd.concat([conjunto_entrenamiento,data],axis=0)
data

#Modelo LAbel Propagation
from sklearn.semi_supervised import LabelPropagation
var_predictoras=data_Total.drop("risk",axis=1)
var_objetivo=data_Total["risk"]
model=LabelPropagation(kernel='knn',n_neighbors=30)
model.fit(var_predictoras,var_objetivo)

"""# 4. Evaluación del modelo sobre el conjunto de prueba
- Exactitud
"""

#Evaluación sobre el conjunto de prueba
from sklearn import metrics

Y_pred = model.predict(X_test)

acc=metrics.accuracy_score(Y_test, Y_pred)
print(f'Exactitud: {acc}')
##LA exctitud es baja debido a la baja cantidad de datos etiquetados

"""# 5. Predicción para datos futuros

- Cargamos los datos futuros
- Aplicamos el modelo para la predicción
"""

#Cargamos los datos futuros
data_fut = pd.read_csv("carRisk-future.csv", sep = ",", na_values = "unknown")
data_fut.head()

#Hacemos la predicción
Y_fut = model.predict(data_fut)
print(Y_fut)