import numpy as np
import scipy as sp

#Initialization
Nx,Ny,Nvecs = 256,256,9
dx
dt
Fi = np.zeros((Nvecs,Nx+2,Ny+2))
FiStar = Fi
rho = np.zeros((Nx,Ny))
u = np.zeros((Nx,Ny))
es = [0,1,-1,1j,-1j,1+1j,-1-1j,-1+1j,1-1j]
# Make the recutangular boundary mask
boundaryMask = np.zeros((Nx,Ny))
boundaryMask[0,:]=boundaryMask[-1,:]=boundaryMask[:,0]=boundaryMask[:,-1]=1
#Streaming
for i in np.arange(Nvecs):
	xShift,yShift = np.real(es[i]),np.imag(es[i])
	xP,xN,yP,yN = xShift*(xShift>0),xShift*(xShift<0),yShift*(yShift>0),yShift*(yShift<0)
	FiStar[i,xP:xN,yP:yN] = Fi[i]
	if i=0 reverseVLayer = 0 else reverseVLayer = !((i-1)%2)+i/2 +1
	FiStar[reverseVLayer][boundaryMask] = FiStar[i]*boundaryMask
	
# Boundary Reflection



rho = np.sum(FiStar,axis=0)
u = (1/rho)*(FiStar*es)*c
