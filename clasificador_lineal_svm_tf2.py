import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plotear = True
################ DATOS ##
#########################
from datasetMocks import linearSVM # modulo propio

all_values = np.array([ row for row in linearSVM ], dtype=np.float32)
x_y_all_values = np.array([ [x, y] for x,y,_z in all_values ], dtype=np.float32)
z_all_values = np.array([ z for _x,_y,z in all_values ], dtype=np.float32)

dataLen = len(x_y_all_values)
_x_y_rows, x_y_columns = x_y_all_values.shape
batch_size = 70

# ######## THE MODEL TRAINING BEGINS #
# ####################################
def perdidaSVMLineal(puntosXY, imagenZ):
	def algoritmo():
		output = tf.reshape(tf.subtract(tf.matmul(puntosXY, pendiente), ordenadaAlOrigen), [-1])

		regularization_loss = tf.divide(tf.square(tf.norm(pendiente, ord=2)), 2)
		hinge_loss = tf.reduce_sum(tf.maximum(0., tf.subtract(1., tf.multiply(imagenZ, output))))
		return tf.add(regularization_loss, tf.multiply(1., hinge_loss))
	return algoritmo

def model_training():
	opt = tf.keras.optimizers.SGD(learning_rate=0.001)

	for _ in range(500):
		rand_idx = np.random.choice(dataLen, size=batch_size)
		rand_xy = x_y_all_values[rand_idx]
		rand_z = np.transpose(z_all_values[rand_idx])
		opt.minimize(perdidaSVMLineal(rand_xy, rand_z), [pendiente, ordenadaAlOrigen])

pendiente = tf.Variable(tf.random.normal(shape=[x_y_columns, 1]))
ordenadaAlOrigen = tf.Variable(tf.random.normal(shape=[1]))
model_training()

# otra manera de resolver el problema de optimización con los mismos resultados:
# regularization_loss = tf.divide(tf.square(tf.norm(pendiente, ord=2)), 2)
# hinge_loss = tf.reduce_sum(tf.minimum(0., tf.subtract(tf.multiply(imagenZ, output), 1.)))
# svm_loss = tf.subtract(tf.reduce_sum(regularization_loss), tf.multiply(1., hinge_loss))

# # ######## TESTEO ##
# # ##################
pend = pendiente.numpy()
ordOrigen = ordenadaAlOrigen.numpy()[0]
calculoSignado = lambda xy: np.sign(np.sum(np.subtract(np.matmul(xy, pend), ordOrigen)))

predicciones = np.transpose(np.array([ calculoSignado(xy) for xy in x_y_all_values ], dtype=np.float32)).flatten()
predYtarget = np.c_[predicciones, z_all_values]
precisionResults = np.array([ z1 == z2 for z1, z2 in predYtarget ], dtype=np.float32)
precision = np.mean(precisionResults)
print("\nPrecición de la prueba de clasificación: {}%".format(str(precision*100)))

# ######## PLOTEO ##
# ##################
if plotear:
	x_objetivo = [ x for x,_y,z in all_values if z ==  1 ]
	y_objetivo = [ y for _x,y,z in all_values if z ==  1 ]
	x_trivial  = [ x for x,_y,z in all_values if z == -1 ]
	y_trivial  = [ y for _x,y,z in all_values if z == -1 ]

	min_x, min_y = np.amin(x_y_all_values,0)
	max_x, max_y = np.amax(x_y_all_values,0)

	pointAmount = 300
	termx = np.linspace(min_x, max_x, pointAmount)
	termy = np.linspace(min_y, max_y, pointAmount)

	xs, ys = np.meshgrid(termx, termy)

	Z = np.transpose(np.array([ calculoSignado(xy) for xy in zip(xs.flatten(), ys.flatten()) ], dtype=np.float32)).flatten()
	Z = Z.reshape(xs.shape)

	plt.suptitle('Linear SVM')
	plt.plot(x_objetivo, y_objetivo, 'ko', label='Objetivo')
	plt.plot(x_trivial, y_trivial, 'kx', label='Trivial')
	plt.contourf(termx, termy, Z, cmap=ListedColormap(['#ad301d', 'green']), alpha=0.8)
	plt.grid(True)
	plt.xlim([0., 1.4])
	plt.ylim([0.8, 4])
	plt.xlabel('Eje X')
	plt.ylabel('Eje Y')

	plt.show()
