# -*- coding: utf-8 -*-
"""Analisis de sentimiento.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WwQ82HIVELS5gskzWbzpyBcj4jfrWM1X

# Clasificación de Texto (Análisis de Sentimiento)
1. Preparar el texto
2. División 70-30
3. Aprendizaje con el 70%
4. Evaluacion del 30%
5. Predicción de datos futuros
"""

#Cargamos librerías principales
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Instalación de paquetes para tratamiento de texto
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud

nltk.download('popular')
stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')

"""#1. Preparación del Texto
- Limpiar el texto
- Eliminar stopwords
- Reducción de raíces
- Visualizaciones del texto
- Representación en vector de características (tfidf)
"""

#Cargamos los datos
data = pd.read_csv("sentimientos.csv",sep = ";", na_values = "unknown", encoding='latin-1')
data.head()

#Convertimos a minuscula y eliminamos caracteres especiales
def tokenizar(texto):
  tokens = word_tokenize(texto)
  words = [w.lower() for w in tokens if w.isalnum()]
  return words
data['tokens'] = data['comentario'].apply(lambda x: tokenizar(x))
data.head()

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

#Reducción a la raíz

def stem_tokens(lista):
  lista_stem = []
  for token in lista:
    lista_stem.append(stemmer.stem(token))
  return lista_stem

data['stemming'] = data['sin_stopwords'].apply(lambda x: stem_tokens(x))
data.head()

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

#Representación en vector de características tf*idf

from sklearn.feature_extraction.text import TfidfVectorizer


def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)  


X = tfidf.fit_transform(data['stemming'])
data_tfidf=pd.DataFrame(X.todense(),columns=tfidf.get_feature_names())
data_tfidf

#Adicionamos el sentimiento a la matriz tfidf
data_tfidf['sentimiento']=data['sentimiento']
data_tfidf.head()

#Graficamos la variable objetivo
data_tfidf['sentimiento'].value_counts().plot(kind='bar')

#Sklearn sólo analiza variables numéricas
data_tfidf["sentimiento"]=data["sentimiento"].replace({"positivo": 1, "negativo": 0})
data_tfidf.head()

"""# 2. División 70-30"""

#División 70-30
from sklearn.model_selection import train_test_split
X = data_tfidf.drop("sentimiento", axis = 1) 
Y = data_tfidf['sentimiento']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
Y_train.value_counts()

"""#3. Aprendizaje del modelo

#A) Arbol
"""

#Creación del modelo con el conjunto de entrenamiento
from sklearn import tree
modelA = tree.DecisionTreeClassifier(min_samples_leaf=2, max_depth=25)
modelA.fit(X_train, Y_train)


from sklearn.externals.six import StringIO 
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
var_predictoras = X.columns.values
nom_clases= ['negativo','positivo']
export_graphviz(modelA, feature_names=var_predictoras, class_names= nom_clases, out_file=dot_data,filled=True, rounded=True,special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

"""# B) Knn"""

#Creación del modelo con el conjunto de entrenamiento
from sklearn import neighbors
modelKnn = neighbors.KNeighborsClassifier(n_neighbors=3, metric='euclidean')
modelKnn.fit(X_train,Y_train)

"""# C)Backpropagation"""

#Creación del modelo con el conjunto de entrenamiento
from sklearn.neural_network import MLPClassifier
modelMLP = MLPClassifier(activation = "relu", hidden_layer_sizes = (25,13), 
                     learning_rate = "adaptive", learning_rate_init = 0.01,
                     momentum = 0.09, max_iter = 200, verbose = True, random_state = 1)
modelMLP.fit(X_train, Y_train)

"""# D) Deep Learning"""

#Creación del modelo con el conjunto de entrenamiento
from keras.models import Sequential
from keras.layers.core import Dense
#Red neuronal
modelDL = Sequential()
modelDL.add(Dense(25, input_dim = 25, activation = 'sigmoid'))
modelDL.add(Dense(13, activation = 'relu'))
modelDL.add(Dense(1, activation = 'relu')) # Capa de salida
modelDL.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
modelDL.fit(X_train, Y_train, epochs=200)

"""# 4. Evaluación del modelo sobre el conjunto de prueba
- Exactitud

#A) Arbol
"""

#Evaluación sobre el conjunto de prueba
from sklearn import metrics

Y_pred = modelA.predict(X_test)

acc=metrics.accuracy_score(Y_test, Y_pred)
print(f'Exactitud: {acc}')

"""# B) Knn"""

#Evaluación sobre el conjunto de prueba
from sklearn import metrics

Y_pred = modelKnn.predict(X_test)

acc=metrics.accuracy_score(Y_test, Y_pred)
print(f'Exactitud: {acc}')

"""# C)Backpropagation"""

#Evaluación sobre el conjunto de prueba
from sklearn import metrics

Y_pred = modelMLP.predict(X_test)

acc=metrics.accuracy_score(Y_test, Y_pred)
print(f'Exactitud: {acc}')

"""# D) Deep Learning"""

#Evaluación sobre el conjunto de prueba
from sklearn import metrics

print(f'Exactitud: {modelDL.evaluate(X_test,Y_test)}') #Exclusiva de Redes neuronales

"""# 5. Predicción para datos futuros
- Cargamos los datos futuros
- Preparamos los datos futuros y creamos tfidf
- Aplicamos el modelo para la predicción
"""

#Cargamos los datos futuros
data_fut = pd.read_csv("sentimientos - test.csv", sep = ";", na_values = "unknown")
data_fut.head()

#Convertimos a minuscula y eliminamos caracteres especiales
def tokenizar(texto):
  tokens = word_tokenize(texto)
  words = [w.lower() for w in tokens if w.isalnum()]
  return words
data_fut['tokens'] = data_fut['comentario'].apply(lambda x: tokenizar(x))
data_fut.head()

#Eliminamos stopwords
def limpiar_tokens(lista):
  clean_tokens = lista[:]
  
  for token in lista:
    if token in stopwords.words('spanish'):
      clean_tokens.remove(token)
  return clean_tokens

# Limpiamos los tokens
data_fut['sin_stopwords'] = data_fut['tokens'].apply(lambda x: limpiar_tokens(x))
data_fut.head()

#Reducción a la raíz

def stem_tokens(lista):
  lista_stem = []
  for token in lista:
    lista_stem.append(stemmer.stem(token))
  return lista_stem

data_fut['stemming'] = data_fut['sin_stopwords'].apply(lambda x: stem_tokens(x))
data_fut.head()

#Representación en vector de características
from sklearn.feature_extraction.text import TfidfVectorizer


#El tfidf de los datos futuros se debe crear según el diccionario inicial
def dummy_fun(doc):
    return doc

tfidf_fut = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None,
    vocabulary=tfidf.vocabulary_)  # Se debe indicar el diccionario del aprendizae


X = tfidf_fut.fit_transform(data_fut['stemming'] )
data_fut_tfidf=pd.DataFrame(X.todense(),columns=tfidf_fut.get_feature_names())


data_fut_tfidf.head()

"""#A) Arbol"""

#Hacemos la predicción
Y_fut = modelA.predict(data_fut_tfidf)
print(Y_fut.round())

"""# B) Knn"""

#Hacemos la predicción
Y_fut = modelKnn.predict(data_fut_tfidf)
print(Y_fut.round())

"""# C)Backpropagation"""

#Hacemos la predicción
Y_fut = modelMLP.predict(data_fut_tfidf)
print(Y_fut.round())

"""# D) Deep Learning"""

#Hacemos la predicción
Y_fut = modelDL.predict(data_fut_tfidf)
print(Y_fut.round())