"""Trabajo final modelamiento matematico

# Juan Camilo Restrepo Velez    000373886
# Wilder Valencia Ocampo        000375627
# Juan Esteban Herrera          000361408
# Emilio Martinez Rivera        000255600
"""

"""##Librerias"""
import csv
import numpy as np

"""##Lectura de Datos"""
def lector(src, fut):
    with open(src) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ';')
        datos = []
        i = 0
        for row in csv_reader:
            if(i>0):
                datos.append(row)
            i += 1
    data = np.array(datos)
    if fut == False: #Condicion para saber si voy a leer datos futuros, para no realizar un balanceo
        data = data[1819:2219, :] #Tomar una parte para realizar el balanceo
    U = np.array(data[:,0:13], dtype=float) #Matriz con las caracteristicas de los usuarios
    P = np.array(data[:,13:21], dtype=float) #Matriz con las caracteristicas de las peliculas
    Z = np.array(data[:,22], dtype=float) #Vector con el gusto por las peliculas
    return U,P,Z

"""##Calculo de Error"""
def Error(U,P,Z,A):
    err = 0
    K = 1
    for i in range(len(U)):
        tmp = np.matmul(np.matmul(U[i,:],A),np.transpose(P[i,:])) #Funcion regresion logistica
        Zest = 1 / (1 + np.exp(- tmp / K) )
        err += (Zest - Z[i])**2
    return err

"""##Calculo del gradiente"""
def gradErr(U,P,Z,A):
    grad = np.zeros_like(A) #Inicializar gradiente con ceros
    e0 = Error(U,P,Z,A) #Calcular error actual
    h = 0.0001
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i,j] = A[i,j] + h #Cambio de i en toda la matriz a
            ei = Error(U,P,Z,A) #Calcular error por cada derivada parcial
            grad[i,j] = (ei - e0)/h #Derivada parcial
            A[i,j] = A[i,j] - h #Volver a la normalidad A
    return grad, e0

"""##Definir la nueva A"""
def NewA(A,alfa, grad):
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i,j] = A[i,j] - alfa*grad[i,j] #Calculo de la nueva A en la direccion donde disminuye la funcion de error con el gradiente
    return A

"""##Disminuir error"""
def DisminuirErr(U,P,Z,A,N,i):
    grad,e0 = gradErr(U,P,Z,A) #Gradiente y erroe inicial
    ant = e0
    alfa = 0.1
    j = 1
    while True: #do ... while
        A = NewA(A,alfa,grad) #Nueva A con cada cambio de gradiente
        grad,act = gradErr(U,P,Z,A) #Calcular nuevo gradiente y error
        act = round(act,3) #Redondear el error para una mejor visualizacion
        print("Iteración ", i,": ", act," con alfa: ", alfa)
        if i > 1:
            if act >= ant: #Si el error actual es mayor o igual al anterior se disminuye alfa
                alfa = alfa/10
                j +=1 #Aumentar variable para saber saber cuantas veces se quedó en el mismo error consecutivamente
            else: #Como el anterior es mayor al actual se vuelve a iniciar el conteo de j
                j = 1
            if j == 4: #Analizar si hay un estancamiento
                i += 1
                return DisminuirErr(U,P,Z,A,N,i) #Llamar iterativamente el metodo cuando hay un estancamiento
        if i == N or round(act,2) == 0.0: #Finaliza de disminuir A cuando se llega al tope de iteraciones (N) o cuando el error es demasiado bajo
            break
        i +=1 #Aumentar contador
        ant = act #Asignar valor del actual para que en la siguiente iteracion sea el anterior
    return A

"""##Estimacion"""
def Estimacion(Ufut,Pfut,A):
    K = 1
    Zest = []
    for i in range(len(Ufut)):
        tmp = np.matmul(np.matmul(Ufut[i,:],A),np.transpose(Pfut[i,:])) #Funcion regresion logistica
        Aux = 1 / (1 + np.exp(-tmp / K) )
        Zest.append(round(Aux,0)) #Agregar la estimacion del usuario i con la pelicula i
    return Zest

"""##Main"""
U,P,Z = lector('datos_recomendador.csv', False) #Leer datos bases  

nU = len(U[0]) #Numero de variables asociadas al usuario
nP = len(P[0]) #Numero de variables Asociadas a la pelicula

A=np.zeros((nU, nP), np.float) #Matriz de numero de variables de usuario por numero de variables de pelicula, donde cada caracteristica esta asociada del usuario esta asociada a todas las variable de pelicula.
grad,e0 = gradErr(U,P,Z,A) #Calcula gradiente y el error en ese momento
print("Error inicial: ", e0)

N = 100 #Numero de veces a iterar
print("\nDisminuir error")
A = DisminuirErr(U,P,Z,A,N,1) #Disminuir el error con la matriz A

Ufut,Pfut,Zfut = lector('datos_futuros.csv', True) #Leer datos futuros reutilizando lector

Zest = Estimacion(Ufut,Pfut,A) #Realizar la estimacion con los datos futuros
print("\nEstimaciones ",Zest)