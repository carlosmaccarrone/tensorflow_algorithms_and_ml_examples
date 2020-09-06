import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Calcular una recta que se apróxime lo mejor posible a los siguientes puntos:
# (1;0), (2;2), (3;0), (4;2)

# y=ax+b
# regresión por cuadrados mínimos 0=error=Σ(aXi+b-Yi)^2
# solución a=0.4, b=0

x_vals = [1., 2., 3., 4.]
y_vals = [0., 2., 0., 2.]

def calculoError(x_data, y_data):
	def error():
		producto = tf.multiply(a,x_data)
		suma = tf.add(producto,b)
		err = tf.square(suma-y_data) / (2*len(x_vals)) # error cuadrático medio
		# print("(X:{} ; Y:{})\t(A:{};B:{})\terror=".format(x_data, y_data, a.numpy()[0], b.numpy()[0], err.numpy()[0]))
		return err
	return error

def model_training():
	opt = tf.keras.optimizers.SGD(learning_rate=0.025)

	for _epoch in range(1000):
		for (x, y) in zip(x_vals, y_vals): 
			opt.minimize(calculoError(x, y), [a, b])

a = tf.Variable(tf.random.normal(shape=[1]))
b = tf.Variable(tf.random.normal(shape=[1]))
model_training()

# Los valores finales no representan la mejor recta posible pero sí es una sucesión convergente.
print("\nEcuación de la recta: y="+str(round(a.numpy()[0],1))+"x+"+str(round(b.numpy()[0],1))+"\n")

# Evaluación del modelo
x_vals_test = [5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 16., 16., 17., 18., 19., 20., 21., 22., 23., 24.]
y_vals_test = [1.6, 3.6, 1.6, 3.6, 3.2, 5.2, 3.2, 5.2, 4.8, 6.8, 4.8, 6.8, 6.4, 8.4, 6.4, 8.4, 8., 10., 8., 10.]
final_a = tf.Variable(a.numpy()[0])
final_b = tf.Variable(b.numpy()[0])

# Se trata de apróximar, por ello no se puede valúar la función pero, se puede 
# calcular la precisión basandose en que el error debe ser siempre próximo a 0.
@tf.function
def cuadradosMin(eje_x, eje_y, n):
	calc1 = tf.multiply(eje_x, final_a)
	calc2 = tf.add(calc1, final_b)
	calc3 = tf.square(calc2-eje_y) / (2*n)
	calc4 = tf.squeeze( tf.round( calc3 ) )
	calc5 = tf.cast(tf.equal(0.0, calc4), tf.float32)
	return tf.reduce_mean(calc5)

eje_x = [x_vals]
eje_y = [y_vals]
n = len(x_vals)
train_prec = cuadradosMin(eje_x, eje_y, n)
print("Precisión en entrenamiento: "+str(train_prec.numpy()*100)+"%")

eje_x = [x_vals_test]
eje_y = [y_vals_test]
n = len(x_vals_test)
test_prec = cuadradosMin(eje_x, eje_y, n)
print("Precisión en testeo: "+str(test_prec.numpy()*100)+"%")

# Ploteo del training
imagen_recta = []
for i in range(-1,30):
    imagen_recta.append(final_a*i+final_b)
x_vals.extend(x_vals_test)
y_vals.extend(y_vals_test)
plt.plot(x_vals, y_vals, 'ro', label='puntos', alpha=0.8)
for (x, y) in zip(x_vals, y_vals):
    plt.text(x-0.2, y-0.2, '({};{})'.format(x, y))
plt.plot(range(-1,30), imagen_recta, label='recta') 
plt.suptitle('Recta que mejor ajusta', fontsize=18) 
plt.xlabel("Eje x")
plt.ylabel("Eje y")
plt.legend(borderpad=1) 
plt.xlim([-1, 26])
plt.ylim([-1, 12])
plt.grid()
plt.show()
