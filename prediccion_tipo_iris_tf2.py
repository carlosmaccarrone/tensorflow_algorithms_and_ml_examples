import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np

######## INICIALIZACIÓN DE LOS DATOS #
######################################
from sklearn import datasets
iris = datasets.load_iris()
# iris.filename = C:\repoPython\venv\lib\site-packages\sklearn\datasets\data\iris.csv
# está manera de inicializarlos no funciona desde csv

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

x_vals_train = np.array([ row[1:5] for row in toTrainFlowersArray ], dtype=np.float32)
y_vals_train = np.array([ row[0] for row in toTrainFlowersArray ], dtype=np.float32)

x_vals_test = np.array([ row[1:5] for row in toTestFlowersArray ])
y_vals_test = np.array([ row[0] for row in toTestFlowersArray ])

trainArrayLen = len(toTrainFlowersArray)

######## THE MODEL TRAINING BEGINS #
####################################
def calculoError(x_data, y_data):
	def error():
		resultado = tf.add(tf.matmul(x_data, pendiente), term_independiente)
		return tf.square(resultado-y_data) / (2*trainArrayLen)
	return error

def model_training():
	opt = tf.keras.optimizers.SGD(learning_rate=0.025)

	for _ in range(5000):
		rand_idx = np.random.choice(trainArrayLen, size=30)
		rand_x, rand_y = (x_vals_train[rand_idx], np.transpose([y_vals_train[rand_idx]]))
		opt.minimize(calculoError(rand_x, rand_y), [pendiente, term_independiente])

pendiente = tf.Variable(tf.random.normal(shape=[4,1]))
term_independiente = tf.Variable(tf.random.normal(shape=[1,1]))
model_training()

######## PREDICCIONES #
#######################
pend = pendiente.numpy()
indep = term_independiente.numpy()

for idx, (x_vals, y_vals) in enumerate(zip(x_vals_test,y_vals_test)):
	prediccion = np.round(np.add(np.matmul(x_vals, pend), indep))
	print("Predicción #{}: {}\t Tipo de flor:{}".format(idx, int(prediccion), int(y_vals)))

