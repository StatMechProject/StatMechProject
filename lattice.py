import numpy as np
import scipy as sp

#Initialization

# Make the recutangular boundary mask
boundaryMask = np.zeros((512,256))
boundaryMask[0,:]=boundaryMask[-1,:]=boundaryMask[:,0]=boundaryMask[:,-1]=1

class Lattice:
    
    def __init__(self,reflectMesh,Nx=512,Ny=256):
        self.Nx,self.Ny,self.Nvecs = 512,256,9
        self.dx=.001; self.dt=.01
        self.c = self.dx/self.dt
        self.Fi = np.zeros((self.Nvecs,Nx,Ny)); 
        self.FiStar = self.Fi
        self.FiEq = self.Fi
        self.rho = np.zeros((Nx,Ny))
        self.u = np.zeros((Nx,Ny))
        self.s = self.Fi
        self.es = np.array([0,1,-1,1j,-1j,1+1j,-1-1j,-1+1j,1-1j]).reshape((9,1,1))
        self.ws = np.array([4/9.,1/9.,1/9.,1/9.,1/9.,1/36.,1/36.,1/36.,1/36.]).reshape((9,1,1))
        self.reflectMesh = reflectMesh.astype(bool)
        #self.updateRhoAndU()
        
    def stream(self):
        es,Nvecs = self.esself.Nvecs
        #Streaming step
        xShift,yShift = np.real(es),np.imag(es)
        for i in np.arange(Nvecs):
            self.FiStar[i] = self.Fi[i]
            if xShift[i]:
                self.FiStar[i] = np.roll(self.FiStar[i],xShift[i],axis=0)
            if yShift[i]:
                self.FiStar[i] = np.roll(self.FiStar[i],yShift[i],axis=1) 
    
    def reflectOnMesh(self):
        es,reflectMask,FiStar,Nvecs = self.es,self.reflectMesh,self.FiStar,self.Nvecs
        # Reflects at the 1's on the mesh
        xShift,yShift = np.real(es),np.imag(es)
        revDirI = np.array([0,2,1,4,3,6,5,8,7])
        for i in np.arange(1,Nvecs,2):
            swapCopy = np.copy(FiStar[i,reflectMask])
            FiStar[i,reflectMask] = FiStar[revDirI[i],reflectMask]
            FiStar[revDirI[i],reflectMask] = swapCopy
        
    def updateRhoAndU(self):
        es,c,FiStar = self.es,self.c,self.FiStar
        self.rho = np.sum(FiStar,axis=0)
        self.u = (1./self.rho)*np.sum(FiStar*es,axis=0)*c




    # Depreciated, replaced by reflectOnMesh
    def reflectOnBorder(self): 
        es,FiStar,Nvecs = self.es,self.FiStar,self.Nvecs
        #Boundary Reflection
        xShift,yShift = np.real(es),np.imag(es)
        xSIndex = .5*xShift-.5; ySIndex = .5*yShift-.5; #Maps 1=>0 and -1=>-1 for indexes
        revDirI = np.array([0,2,1,4,3,6,5,8,7])
        for i in np.arange(Nvecs): 
            if xShift[i]:
                FiStar[i,xSIndex[i],:]=FiStar[revDirI[i],xSIndex[revDirI[i]],:]
            if yShift[i]:
                FiStar[i,:,ySIndex[i]]=FiStar[revDirI[i],:,ySIndex[revDirI[i]]]

    def updateFi():
        conj_u = np.conj(self.u)
        product = np.real(self.es * conj_u))
        self.s = self.ws * (3/self.c * product + 9/(2 * self.c ** 2) * product ** 2 - 3/(2 * self.c^2) * self.u * conj_u
        self.FiEq = self.ws * self.rho + self.rho * self.s
        self.Fi = self.Fi - 1/self.tau * (self.FiStar - self.FiEq)



class Visualization:
    
    def __init__(self,bLattice):
        self.lattice = bLattice
        
    def update(self,time):
        return
    
    def animate(self):
        return
