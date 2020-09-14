import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

################ CONSTANTES ########
####################################
batch_size = 100 
numNodos, iteraciones = (5, 200)  ## (3, 1000) # según la configuración el modelo servirá más o menos
plotear = True

################ DATOS ##
#########################
(x_y_all_values, z_all_values) = datasets.make_moons(n_samples=50, noise=.1) ## switch a medialunas
# (x_y_all_values, z_all_values) = datasets.make_circles(n_samples=500, factor=0.5, noise=0.1) ## switch a circunferencias

x_y_values = np.array(x_y_all_values, dtype=np.float32)
zn_z_values = np.array([ [0,1] if z else [1,0] for z in z_all_values ], dtype=np.float32)

_x_y_rows, x_y_columns = x_y_values.shape
_zn_z_rows, zn_z_columns = zn_z_values.shape

# ######## THE MODEL TRAINING BEGINS #
# ####################################
def modeloDeDosCapas(puntosXY):
	calc1 = tf.matmul(puntosXY, pendiente_1)
	calc2 = tf.add(calc1, ordenada_1)
	primeraCapa = tf.nn.tanh(calc2)

	calc3 = tf.matmul(primeraCapa, pendiente_2)
	calc4 = tf.add(calc3, ordenada_2)
	return tf.nn.softmax(calc4)

def errorDeEntropiaCruzada(imagen, imagenZNZ):
	calc5 = tf.math.log(imagen)
	calc6 = tf.multiply(imagenZNZ, calc5)
	calc7 = tf.reduce_sum(calc6)
	return tf.negative(calc7)

def algoritmo(puntosXY, imagenZNZ):
	def wrap():
		imagen = modeloDeDosCapas(puntosXY)
		return errorDeEntropiaCruzada(imagen, imagenZNZ)
	return wrap

def model_training():
	opt = tf.keras.optimizers.SGD(learning_rate=0.01)

	for _ in range(iteraciones):
		rand_idx = np.random.choice(len(x_y_values), size=batch_size)
		puntosXY = x_y_values[rand_idx]
		imagenZNZ = zn_z_values[rand_idx]
		opt.minimize(algoritmo(puntosXY, imagenZNZ), [pendiente_1, ordenada_1, pendiente_2, ordenada_2])

pendiente_1 = tf.Variable(tf.random.normal(shape=[x_y_columns, numNodos]))
pendiente_2 = tf.Variable(tf.random.normal(shape=[numNodos, zn_z_columns]))
ordenada_1 = tf.Variable(tf.zeros([1, numNodos]))
ordenada_2 = tf.Variable(tf.zeros([1, zn_z_columns]))
model_training()

# # ######## TESTEO ##
# # ##################
pend1 = pendiente_1.numpy()
pend2 = pendiente_2.numpy()
ord1 = ordenada_1.numpy()
ord2 = ordenada_2.numpy()

def clasificarPuntos(puntosXY):
	calc1 = np.matmul(puntosXY, pend1)
	calc2 = np.add(calc1, ord1)
	primeraCapa = np.tanh(calc2)

	calc3 = np.matmul(primeraCapa, pend2)
	calc4 = np.add(calc3, ord2)
	imagen = np.divide(np.exp(calc4), np.sum(np.exp(calc4))) # softmax function
	return np.argmax(imagen, 1)

imagenResultante = clasificarPuntos(x_y_values)
predicciones = np.equal(imagenResultante, np.argmax(zn_z_values, 1))
precision = np.mean(np.array(predicciones, dtype=np.float32))
print("\nPrecisión: {}".format(str(precision*100)))

# ######## PLOTEO ##
# ##################
if plotear:
	min_x, min_y = np.amin(x_y_values,0)-1
	max_x, max_y = np.amax(x_y_values,0)+1

	pointAmount = 300
	termx = np.linspace(min_x, max_x, pointAmount)
	termy = np.linspace(min_y, max_y, pointAmount)

	xs, ys = np.meshgrid(termx, termy)

	contents = np.c_[x_y_all_values, z_all_values]
	x_circle  = [ vec[0] for vec in contents if vec[2] == 1 ]
	y_circle  = [ vec[1] for vec in contents if vec[2] == 1 ]
	x_cross   = [ vec[0] for vec in contents if vec[2] == 0 ]
	y_cross   = [ vec[1] for vec in contents if vec[2] == 0 ]

	Z = clasificarPuntos(np.c_[xs.flatten(), ys.flatten()]);
	Z = Z.reshape(xs.shape)

	fig = plt.figure(figsize=(12,5))
	fig.suptitle('Softmax non-linear classifications')
	ax1 = fig.add_subplot(1, 2, 1)

	ax1.plot(x_circle, y_circle, 'ko')
	ax1.plot(x_cross, y_cross, 'kx')    
	ax1.contourf(xs, ys, Z, cmap=ListedColormap(['#ad301d', 'g']))
	ax1.set_xlabel('Eje X')
	ax1.set_ylabel('Eje Y')
	
	ax2 = fig.add_subplot(1, 2, 2, projection='3d')
	ax2.scatter3D(x_y_values[:,0], x_y_values[:,1], c=zn_z_values[:,1], s=50, cmap=ListedColormap(['red', 'green']))

	plt.show()
