import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.disable_v2_behavior()
session = tf.Session()

import numpy as np
import math
sigmoide = lambda x: 1 / (1 + math.exp(-x))
batch_size = 30

######## INICIALIZACIÓN DE LOS DATOS #
######################################
from sklearn import datasets
iris = datasets.load_iris()
# iris.filename = C:\repoPython\venv\lib\site-packages\sklearn\datasets\data\iris.csv
# ésta manera de inicializarlos no funciona desde csv

setosa = np.array([ data for idx, data in enumerate(iris.data) if idx <= 49])
versicolor = np.array([ data for idx, data in enumerate(iris.data) if idx > 49 and idx <= 99])
virginica = np.array([ data for idx, data in enumerate(iris.data) if idx > 99])
assert (setosa[0:50] == iris.data[0:50]).all()
assert (versicolor[0:50] == iris.data[50:100]).all()
assert (virginica[0:50] == iris.data[100:150]).all()

setosaVals = np.insert(setosa, 0, 0, axis=1)
versicolorVals = np.insert(versicolor, 0, 1, axis=1)
virginicaVals = np.insert(virginica, 0, 2, axis=1)

flowersArray = np.concatenate((setosaVals, versicolorVals, virginicaVals))

data_values = np.array([ row[1:5] for row in flowersArray ], dtype=np.float32)
label_values = np.array([ row[0] for row in flowersArray ], dtype=np.float32)
dataLen = len(flowersArray)

######## THE MODEL TRAINING BEGINS #
####################################
x_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

pend = tf.Variable(tf.random_normal(shape=[4,1]))
ordenadaAlOrigen = tf.Variable(tf.random_normal(shape=[1,1]))
resultado = tf.add(tf.matmul(x_data, pend), ordenadaAlOrigen)

cuadraticError = tf.divide(tf.square(tf.subtract(resultado, y_data)), tf.multiply(2., dataLen))
# para obtener mayor precision es mejor no reducir los valores.

init = tf.global_variables_initializer()
session.run(init)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cuadraticError)

for _ in range(300):
	rand_idx = np.random.choice(dataLen, size=batch_size)
	rand_x, rand_y = (data_values[rand_idx], np.transpose([label_values[rand_idx]]))
	session.run(optimizer, feed_dict={x_data: rand_x, y_data: rand_y})

######## PREDICCIONES #
#######################
pendiente = session.run(pend)
ordenada = session.run(ordenadaAlOrigen)

predicciones = np.array([ np.round(np.add(np.matmul(x, pendiente), ordenada)) for x in data_values ], dtype=np.float32).flatten()
for idx in range(len(label_values)):
	print("Predicción #{}: {}\t Tipo de flor:{}".format(idx, int(predicciones[idx]), int(label_values[idx])))

######## BONUS #
################
# # # # Si uno quiere estimar la probabilidad de un resultado o regresión logística donde 
# # # # la clasificación debe ser binaria, puede usarse la función sigmoide redondeada.
errorCuadratico = np.divide(np.square(np.subtract(predicciones, np.transpose(label_values))), np.multiply(2., dataLen))
mediaError = np.mean(errorCuadratico)
sigmoideRedondeada = np.round(sigmoide(mediaError))
probabilidades = np.array(np.equal(sigmoideRedondeada, label_values), dtype=np.float32)
probabilidad = np.mean(probabilidades)

print("\nHay un porcentaje de probabilidades de que una flor de iris dada según\
	   sus sépalos y pétalos pertenesca a una especie en particular, como\
	   tenemos tres conjuntos de datos de igual tamanio y cada uno representa\
	   a una especie lo normal es que la probabilidad sea una de tres o 33%\
	   de probabilidades que pertenesca a una especie y no a las otras dos")
print("Probabilidad: {}".format(probabilidad))

contents = np.c_[predicciones, label_values]
precision = np.mean(np.array([ dat[0] == dat[1] for dat in contents ], dtype=np.float32))
print("Presición del algoritmo: ", precision)
