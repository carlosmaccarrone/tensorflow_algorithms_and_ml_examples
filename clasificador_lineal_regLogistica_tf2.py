import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plotear = True
################ DATOS ##
#########################
from datasetMocks import linearSoft # modulo propio

all_values = np.array([ row for row in linearSoft ], dtype=np.float32)
x_y_values = np.array([ [x, y] for x,y,_z in all_values ], dtype=np.float32)
z_values = np.array([ z for _x,_y,z in all_values ], dtype=np.float32)

_x_y_rows, x_y_columns = x_y_values.shape
dataLen = len(x_y_values)
batch_size = 100

# ######## THE MODEL TRAINING BEGINS #
# ####################################
def modeloLineal(puntosXY):
	calc1 = tf.matmul(puntosXY, pendiente)
	calc2 = tf.add(calc1, ordenadaAlOrigen)
	return calc2

def algoritmo(puntosXY, imagenZ):
	def wrap():
		imagen = tf.reshape(modeloLineal(puntosXY), [-1])
		entropia_cruzada = tf.nn.sigmoid_cross_entropy_with_logits(logits=imagen, labels=imagenZ)
		return tf.reduce_mean(entropia_cruzada)
	return wrap

def model_training():
	opt = tf.keras.optimizers.SGD(learning_rate=0.01)

	for _ in range(300):
		rand_idx = np.random.choice(len(x_y_values), size=batch_size)
		puntosXY = x_y_values[rand_idx]
		imagenZ = z_values[rand_idx]
		opt.minimize(algoritmo(puntosXY, imagenZ), [pendiente, ordenadaAlOrigen])

pendiente = tf.Variable(tf.zeros([x_y_columns, 1]))
ordenadaAlOrigen = tf.Variable(tf.zeros([1]))
model_training()

# ######## TESTEO ##
# ##################
pend = pendiente.numpy()
ordenada = ordenadaAlOrigen.numpy()[0]
sigmoide = lambda x: 1 / (1 + np.exp(-x))

def clasificarPuntos(puntosXY):
	calc1 = np.matmul(puntosXY, pend)
	calc2 = np.add(calc1, ordenada)
	output = sigmoide(calc2)
	return np.round(output).flatten()

imagenResultante = clasificarPuntos(x_y_values)
predicciones = np.equal(imagenResultante, z_values)
precision = np.mean(np.array(predicciones, dtype=np.float32))

for i in range(dataLen):
	print( "Predicción {} == ImagenZ {} : {}".format(imagenResultante[i], z_values[i], predicciones[i]) )

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

	contents = np.c_[x_y_values, z_values]
	x_circle  = [ vec[0] for vec in contents if vec[2] == 1 ]
	y_circle  = [ vec[1] for vec in contents if vec[2] == 1 ]
	x_cross   = [ vec[0] for vec in contents if vec[2] == 0 ]
	y_cross   = [ vec[1] for vec in contents if vec[2] == 0 ]

	Z = clasificarPuntos(np.c_[xs.flatten(), ys.flatten()]);
	Z = Z.reshape(xs.shape)

	plt.suptitle('Logistic regression classifications')

	plt.plot(x_circle, y_circle, 'ko')
	plt.plot(x_cross, y_cross, 'kx')    
	plt.contourf(xs, ys, Z, cmap=ListedColormap(['#ad301d', 'g']), alpha=0.8)
	plt.grid(True)
	plt.xlabel('Eje X')
	plt.ylabel('Eje Y')
	plt.xlim([-1.1, 2.5])
	plt.ylim([-0.3, 5.2])

	plt.show()
	