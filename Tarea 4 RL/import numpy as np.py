import numpy as np
import matplotlib.pyplot as plt

# Definimos el dominio
x_vals = np.linspace(-1, 1, 400)

# Calculamos las funciones objetivo
f1 = np.sqrt(5 - x_vals**2)
f2 = x_vals / 2

# Graficamos el espacio de decisión
plt.figure()
plt.plot(x_vals, f1, label=r'$f_1(x) = \sqrt{5 - x^2}$')
plt.plot(x_vals, f2, label=r'$f_2(x) = \frac{x}{2}$')
plt.title("Espacio de decisión")
plt.xlabel("x")
plt.ylabel("f_i(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graficamos el espacio de objetivos
plt.figure()
plt.plot(f1, f2, label="Curva objetivo", color='blue')
plt.title("Espacio de objetivos")
plt.xlabel(r'$f_1(x) = \sqrt{5 - x^2}$')
plt.ylabel(r'$f_2(x) = \frac{x}{2}$')
plt.grid(True)
plt.tight_layout()
plt.show()
