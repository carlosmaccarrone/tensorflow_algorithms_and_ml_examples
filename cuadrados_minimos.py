import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
session = tf.Session()
import tensorflow.compat.v1.nn as nn

import numpy as np
import matplotlib.pyplot as plt

# Calcular una recta que se apróxime lo mejor posible a los siguientes puntos:
# (1;0), (2;2), (3;0), (4;2)

# y=ax+b
# regresión por cuadrados mínimos 0=error=Σ(aXi+b-Yi)^2
# solución a=0.4, b=0

x_vals = [1, 2, 3, 4]
y_vals = [0, 2, 0, 2]

x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_data = tf.placeholder(shape=[1], dtype=tf.float32)
a = tf.Variable(tf.random_normal(shape=[1]))
b = tf.Variable(tf.random_normal(shape=[1]))

producto = tf.multiply(a,x_data)
suma = tf.add(producto,b)

error = tf.square(suma-y_data) / (2*len(x_vals)) #el error hace la diferencia
# error = tf.square(suma-y_data)

optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.025).minimize(error)

init = tf.global_variables_initializer()
session.run(init)

for epoch in range(1000):
	for (x, y) in zip(x_vals, y_vals): 
		session.run( optimizador, feed_dict={x_data: [x], y_data: [y]} )

	# print( "a="+str(session.run(a))+", b="+str(session.run(b)) )

final_a = session.run(a)
final_b = session.run(b)

# Los valores finales no representan la mejor recta posible pero sí es una sucesión convergente.
print("\nEcuación de la recta: y="+str(np.round(final_a,1)[0])+"x+"+str(np.round(final_b,1)[0])+"\n")

imagen_recta = []
for i in range(-1,7):
    imagen_recta.append(final_a*i+final_b)

plt.plot(x_vals, y_vals, 'ro', label='puntos') 
plt.plot(range(-1,7), imagen_recta, label='recta') 
plt.suptitle('Recta que mejor ajusta', fontsize=18) 
plt.xlabel("Eje x")
plt.ylabel("Eje y")
plt.legend(borderpad=1) 
plt.xlim([-1, 6])
plt.ylim([-1, 3])
plt.show()
