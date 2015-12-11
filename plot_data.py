import numpy as np
import matplotlib.pyplot as plt

simType = raw_input('Please give the file name for the run: ')
ux = np.loadtxt(simType + '_ux.csv')
uy = np.loadtxt(simType + '_uy.csv')
rho = np.loadtxt(simType + '_rho.csv')

plt.imshow(np.flipud(ux.T))
plt.title('Flat Plate X-Direction Velocity', size = 14)
cbar = plt.colorbar()
cbar.set_label('m/s',size = 12)
plt.axis('off')
plt.savefig(simType + '.pdf', bbox_inches='tight')
plt.show()
