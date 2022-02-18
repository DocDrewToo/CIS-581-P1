import numpy as np
import matplotlib.pyplot as plt

a = 3
b = 4
c = 2
x = np.linspace(0, 10, 256, endpoint = True)
y = (a * (x * x)) + (b * x) + c

plt.plot(x, y, '-g', label=r'$y = 3x^2 + 4x + 2$')

axes = plt.gca()
axes.set_xlim([x.min(), x.max()])
axes.set_ylim([y.min(), y.max()])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Curve')
plt.legend(loc='upper left')

plt.show()