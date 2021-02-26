#########################################################################################

# Limpiar workspace
rm(list = ls())

# Cargar datos
datos <- read.csv(file = file.choose(), header = TRUE, sep = ",")
#########################################################################################

# PREPARACION DE DATOS
#########################################################################################
# Histograma
hist(datos$age, main = 'Age')
hist(datos$campaign, main = 'campaign')
hist(datos$previous, main = 'previous')
hist(datos$cons.price.idx, main = 'cons.price.idx')
hist(datos$cons.conf.idx, main = 'cons.conf.idx')
hist(datos$nr.employed, main = 'nr.employed')
# Grafico de barras
barplot(table(datos$job), main = 'job')
barplot(table(datos$marital), main = 'marital')
barplot(table(datos$education), main = 'education')
barplot(table(datos$default), main = 'default')
barplot(table(datos$housing), main = 'housing')
barplot(table(datos$loan), main = 'loan')
barplot(table(datos$month), main = 'month')
barplot(table(datos$day_of_week), main = 'day_of_week')
barplot(table(datos$y), main = 'y')

#########################################################################################

# DIVISION 70 - 30
#########################################################################################
library(caret)

intrain <- createDataPartition(datos$y, p = 0.7, list = FALSE)
training <- datos[intrain,]
testing <- datos[-intrain,]
#########################################################################################

# ARBOL DE DESICION
#########################################################################################
library(rpart)
library(rpart.plot)

modelo <- rpart(y ~ ., method = 'class', data = training, control = rpart.control(minbucket=5, maxdepth=7))
windows()
rpart.plot(modelo, type = 5)
#########################################################################################

# Evaluacion
#########################################################################################
library(e1071)

predicciones <- predict(modelo, testing, type = 'class')
confusionMatrix(data = as.factor(predicciones), reference = as.factor(testing$y), mode = 'prec_recall')

# Guardar el modelo
save(modelo, file="C:/Users/Administrador/Downloads/Arbol.rmodel")
#########################################################################################

# Prediccion futura
#########################################################################################
rm(list = ls())

# Cargar el modelo
load(file.choose())

#Cargar datos
datosFuturos <- read.csv(file = file.choose(), header = TRUE, sep = ',')

prediccion <- predict(modelo, datosFuturos, type = 'class')
print(prediccion)
#########################################################################################