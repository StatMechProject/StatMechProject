import lattice
import numpy as np

simType = raw_input('Please give the file name for the run: ')
x = lattice.Lattice()

for i in range(10000):
    x.fullTimeStep()
    
    if i % 10 == 0:
        print x.convergence()
        np.savetxt(simType + '_ux.csv', np.real(x.u))
        np.savetxt(simType + '_uy.csv', np.imag(x.u))
        np.savetxt(simType + '_rho.csv', x.rho)
        
