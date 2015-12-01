import Lettuce
import matplotlib.pyplot as plt
import numpy as np
x = Lettuce.Lattice()

for i in range(1000000):
	x.fullTimeStep()
	if i % 5000 == 0:
		plt.imshow(np.abs(x.u))
		plt.colorbar()
		plt.show()

