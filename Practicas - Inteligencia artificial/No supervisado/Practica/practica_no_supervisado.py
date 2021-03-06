# -*- coding: utf-8 -*-
"""Practica - No Supervisado

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iu8S8XoCMzkATIdtfrZzwcT7OEYLqQ4V

# Juan Camilo Restrepo Velez - 000373886
"""

#Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""# 1. Preparación de Datos
-  Cargamos los datos
-  Conocemos los datos con estadísticos
- Transformación de datos  (k-means sólo analiza variables numéricas)
"""

#Cargamos los datos
data = pd.read_excel("Aprobación curso-InteligenciaArtificial1.xlsx", sheet_name=0)
data.head()

#Conocemos los datos
data.info()

#Corrección del tipo de datos
data['Felder']=data['Felder'].astype('category')
data.info()

#Descripción de variables numéricas
print(data.describe());

#Descripción variables categóricas
data['Felder'].value_counts().plot(kind='bar')

#Seleccion de variables
data = data.drop('ID', axis=1) #Eliminar el ID por ser irrelevante
data.info();

#Limpiamos datos atipicos
data.Examen_admision[data["Examen_admision"]<3]= None #Poner en nulo los datos que sean menor a 3 de la columna Examen admision
data.info();

#Limpieza de nulos: Imputacion por media
data['Examen_admision'] = data['Examen_admision'].fillna(value = data['Examen_admision'].mean()) #Poner la media en los datos nulos de la columna
data.info()

#Se codifican las categorias de las variables
dummiesFelder = pd.get_dummies(data['Felder']) 
data = data.drop('Felder', axis=1)
data = data.join(dummiesFelder)
data.head()

"""# 2. Aprendizaje del Modelo: K-means"""

from sklearn.cluster import KMeans
model = KMeans(n_clusters = 7, max_iter = 100) #Se escogen 7 cluster debido al metodo del codo
model.fit(data)

"""#Saber el numero de clusters

Método del codo
"""

#Metodo del codo para encontrar la menor cantidad de clusters
ks=range(1,20) #crear valores del 1 al< 5
inertias=[]

for k in ks:
  #crear modelo
  model=KMeans(n_clusters=k)
  model.fit(data)
  inertias.append(model.inertia_)
#graficar cantidad de clusters vs inertias
plt.plot(ks,inertias,'-o')
plt.xlabel('Numero de clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

"""# 3. Evaluación del modelo 
- Cohesion 
- Silueta(Inercia del modelo)
"""

#Inercia del modelo
print(f'inertia del modelo= {model.inertia_}')

#Evalucaion:Silueta
from sklearn import metrics
sil=metrics.silhouette_score(data,model.predict(data))
print(f'Indice de Silueta={sil}') #Como la silueta dio mayor a 0.5 se dejan los cluster elegidos

"""# 4. Perfilamiento"""

#Centroides de los clusters
centroides = pd.DataFrame(model.cluster_centers_, columns = data.columns.values)
centroides.round(0)

"""- **0: Visual ->** Se hace lo que se puede, se pasó el examen de admisión en 3 al igual que la nota final del curso
- **1: Sensorial ->** Salvando la nota, pasando en 3 todo
- **2: Secuencial ->** Lo importante es pasarla, si saca 3 en la nota de admision saca 3 en la nota final
- **3: Equilibrio ->** Si aprendieron, pasando a la u en 3 y aprobando el curso en 4
- **4: Des-Equilibrio ->** No aprendieron, se logró pasar en 3 pero perdieron en 2 el curso
- **5: Activo ->** Activo pero no para el estudio, a duras se pasa en 3 la admision y terminan el curso en 2
- **6: Casi-Visual ->** Se intentó pero no se logró, sacaron el examen de admision en 3 y el curso en 2
"""