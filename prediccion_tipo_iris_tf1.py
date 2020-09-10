import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.disable_v2_behavior()
session = tf.Session()

import numpy as np

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

toTrainFlowersArray = np.concatenate((setosaVals[0:40], versicolorVals[0:40], virginicaVals[0:40]))
toTestFlowersArray = np.concatenate((setosaVals[40:50], versicolorVals[40:50], virginicaVals[40:50]))

x_vals_train = np.array([ row[1:5] for row in toTrainFlowersArray ])
y_vals_train = np.array([ row[0] for row in toTrainFlowersArray ])

x_vals_test = np.array([ row[1:5] for row in toTestFlowersArray ])
y_vals_test = np.array([ row[0] for row in toTestFlowersArray ])

trainArrayLen = len(toTrainFlowersArray)

######## THE MODEL TRAINING BEGINS #
####################################
batch_size = 30
x_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

pendiente = tf.Variable(tf.random_normal(shape=[4,1]))
term_independiente = tf.Variable(tf.random_normal(shape=[1,1]))
resultado = tf.add(tf.matmul(x_data, pendiente), term_independiente)
error = tf.square(resultado-y_data) / (2*trainArrayLen)

init = tf.global_variables_initializer()
session.run(init)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

for _ in range(5000):
	rand_idx = np.random.choice(trainArrayLen, size=batch_size)
	rand_x, rand_y = (x_vals_train[rand_idx], np.transpose([y_vals_train[rand_idx]]))
	session.run(optimizer, feed_dict={x_data: rand_x, y_data: rand_y})

######## PREDICCIONES #
#######################
pend = session.run(pendiente)
indep = session.run(term_independiente)

for idx, (x_vals, y_vals) in enumerate(zip(x_vals_test,y_vals_test)):
	prediccion = np.round(np.add(np.matmul(x_vals, pend), indep))
	print("Predicción #{}: {}\t Tipo de flor:{}".format(idx, int(prediccion), int(y_vals)))

######## BONUS #
################
# # # # Si uno quiere predecir la probabilidad de un resultado o regresión logística donde 
# # # # la clasificación debe ser binaria, puede usarse la función sigmoide redondeada.
sigmoideRedondeada = tf.round(tf.sigmoid(error))
probabilidades = tf.cast(tf.equal(sigmoideRedondeada, y_data), tf.float32)
probabilidad = tf.reduce_mean(probabilidades)

trainin_prob = session.run(probabilidad, feed_dict={x_data: x_vals_train, y_data: np.transpose([y_vals_train])})
testing_prob = session.run(probabilidad, feed_dict={x_data: x_vals_test, y_data: np.transpose([y_vals_test])})

print("\nHay un porcentaje de probabilidades de que una flor de iris dada según\
	   sus sépalos y pétalos pertenesca a una especie en particular, como\
	   tenemos tres conjuntos de datos de igual tamanio y cada uno representa\
	   a una especie lo normal es que la probabilidad sea una de tres o 33%\
	   de probabilidades que pertenesca a una especie y no a las otras dos")
print("Probabilidad en el conjunto de entrenamiento: {}".format(trainin_prob))
print("Probabilidad en el conjunto de pruebas: {}".format(testing_prob))

