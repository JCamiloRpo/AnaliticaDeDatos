# -*- coding: utf-8 -*-
"""trenesValencia.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IRMuVUGcP27V3G1tZLL2NW7WSPLMDmxh

# Preparación de Texto 
1. Limpieza del Texto
2. Eliminar Stopwords
3. Reducir las palabras a  las raíces
4. Visualización del texto
"""

#Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Instalación de paquetes para tratamiento de texto
import nltk #Libreria para procesamiento de lenguaje natural
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud

nltk.download('popular')
stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')

"""#1. Limpieza del Texto
- Cargar los datos
- Eliminar números y caracteres especiales
- Convertir a minúscula
"""

#Cargamos los datos
data = pd.read_csv("trenesValencia.csv", sep = ",", na_values = "unknown")
data.head()

data.info()

#Convertimos a minuscula y eliminamos caracteres especiales
def tokenizar(texto):
  tokens = word_tokenize(texto)
  words = [w.lower() for w in tokens if w.isalnum()]
  return words
data['tokens'] = data['Llamada'].apply(lambda x: tokenizar(x))
data.head()

"""#2. Eliminamos stopwords en español"""

#Eliminamos stopwords
def limpiar_tokens(lista):
  clean_tokens = lista[:]
  sr = stopwords.words('spanish')
  for token in lista:
    if token in stopwords.words('spanish'):
      clean_tokens.remove(token)
  return clean_tokens

# Limpiamos los tokens
data['sin_stopwords'] = data['tokens'].apply(lambda x: limpiar_tokens(x))
data.head()

"""#3. Reducción de raíces en español"""

#Reducción a la raíz

def stem_tokens(lista):
  lista_stem = []
  for token in lista:
    lista_stem.append(stemmer.stem(token))
  return lista_stem

data['stemming'] = data['sin_stopwords'].apply(lambda x: stem_tokens(x))
data.head()

"""#4. Visualizaciones del texto"""

#Nube de palabras 
lista_palabras = data["stemming"].tolist()
tokens_abstractos = [keyword.strip() for sublista in lista_palabras for keyword in sublista]
wordcloud = WordCloud( max_words=1000,margin=0).generate((" ").join(tokens_abstractos))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#Gráfica de palabras mas frecuentes
freq = nltk.FreqDist(tokens_abstractos)
plt.figure(figsize=(8, 8))
freq.plot(20, cumulative=False)