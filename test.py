import lattice
import matplotlib.pyplot as plt
import numpy as np
x = lattice.Lattice()

for i in range(1000000):
	x.fullTimeStep()
	if i % 200 == 0:
		fig = plt.imshow(np.real(x.u).T)
  		plt.title('Cylinder X-Direction Velocity', size = 14)
		cbar = plt.colorbar()
  		cbar.set_label('m/s',size = 12)
  		plt.axis('off')
		plt.savefig('test.pdf', bbox_inches='tight')
		plt.show()
		print x.convergence()
