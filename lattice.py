import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Make the recutangular boundary mask
boundaryMask = np.zeros((512,256))
X,Y  = np.mgrid[0:512,0:256]
#boundaryMask[0,:]=boundaryMask[-1,:]=
boundaryMask[:,0]=boundaryMask[:,-1]=1
circleMask = ((256-X)**2 + (128-Y)**2) < 25**2
boundaryMask[circleMask] = 1



class Lattice:
    
    def __init__(self,reflectMesh=boundaryMask,Nx=512,Ny=256):
    	self.mew = .00001857 # constant for the mu
        self.Nx,self.Ny,self.Nvecs = 512,256,9
        self.dx=.001; self.dt=.01
        self.c = self.dx/self.dt
        self.Fi = np.zeros((self.Nvecs,Nx,Ny)); 
        self.FiStar = self.Fi
        self.FiEq = self.Fi
        self.rho = np.zeros((Nx,Ny))
        self.u = np.zeros((Nx,Ny))
        self.s = self.Fi
        self.es = np.array([0,1,1j,-1,-1j,1+1j,-1+1j,-1-1j,1-1j]).reshape((9,1,1))
        self.ws = np.array([4/9.,1/9.,1/9.,1/9.,1/9.,1/36.,1/36.,1/36.,1/36.]).reshape((9,1,1))
        self.reflectMesh = reflectMesh.astype(bool)
        self.inletVelocity = 25
        #self.updateRhoAndU()
        self.initFs()
        

    def initFs(self):
        # using density of air as 1.225 kg/m3 so over 1/800g per mm2
        self.Fi = np.ones((self.Nvecs,self.Nx,self.Ny))/800.
        self.stream()
        self.updateRhoAndU()

        
    def stream(self):
        es,Nvecs = self.es,self.Nvecs
        #Streaming step
        xShift,yShift = np.real(es).astype(int),np.imag(es).astype(int)
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
        revDirI = np.array([0,3,4,1,2,7,8,5,6])
        for i in np.arange(1,Nvecs,2):
            swapCopy = np.copy(FiStar[i,reflectMask])
            FiStar[i,reflectMask] = FiStar[revDirI[i],reflectMask]
            FiStar[revDirI[i],reflectMask] = swapCopy
        
    def updateRhoAndU(self):
        es,c,FiStar = self.es,self.c,self.FiStar
        self.rho = np.sum(FiStar,axis=0)
        self.u = (1./self.rho)*np.sum(FiStar*es,axis=0)*c



    def vorticity(self):
        # single sided finite difference for del x u (curl u)
        ux,uy = np.real(self.u),np.imag(self.u)
        uxincy,uyincx = np.roll(ux,1,1),np.roll(uy,1,0)
        curlU = (uxincy-ux  -uyincx+uy)/self.dx
        return curlU



    def inletOutlet(self):
        FiStar = self.FiStar
        # inlet
        self.rho[0,:] = (FiStar[0,0,:]+FiStar[2,0,:]+FiStar[4,0,:]+\
                   2*(FiStar[3,0,:]+FiStar[6,0,:]+FiStar[7,0,:]))/(1-self.inletVelocity)
        FiStar[1,0,:] = FiStar[3,0,:]
        FiStar[5,0,:] = FiStar[7,0,:] - 1/2 * (FiStar[2,0,:] - FiStar[4,0,:]) + 1/6 * self.rho[0,:] * self.inletVelocity
        FiStar[8,0,:] = FiStar[6,0,:] - 1/2 * (FiStar[2,0,:] - FiStar[4,0,:]) + 1/6 * self.rho[0,:] * self.inletVelocity
        
        # outlet
        # done by swapping respective 3-1, 6-5, 7-8 for all following code
        self.rho[-1,:] = (FiStar[0,0,:]+FiStar[2,0,:]+FiStar[4,0,:]+\
                    2*(FiStar[1,0,:]+FiStar[5,0,:]+FiStar[8,0,:]))/(1-self.inletVelocity)
        FiStar[3,0,:] = FiStar[1,0,:]
        FiStar[6,0,:] = FiStar[5,0,:] - 1/2 * (FiStar[2,0,:] - FiStar[4,0,:]) + 1/6 * self.rho[0,:] * self.inletVelocity
        FiStar[7,0,:] = FiStar[8,0,:] - 1/2 * (FiStar[2,0,:] - FiStar[4,0,:]) + 1/6 * self.rho[0,:] * self.inletVelocity




    # Depreciated, replaced by reflectOnMesh
    def reflectOnBorder(self): 
        es,FiStar,Nvecs = self.es,self.FiStar,self.Nvecs
        #Boundary Reflection
        xShift,yShift = np.real(es),np.imag(es) #Maps 1=>0 and -1=>-1 for indexes
        xSIndex = (.5*xShift-.5).astype(int); ySIndex = (.5*yShift-.5).astype(int); 
        revDirI = np.array([0,3,4,1,2,7,8,5,6])
        for i in np.arange(Nvecs): 
            if xShift[i]:
                FiStar[i,xSIndex[i],:]=FiStar[revDirI[i],xSIndex[revDirI[i]],:]
            if yShift[i]:
                FiStar[i,:,ySIndex[i]]=FiStar[revDirI[i],:,ySIndex[revDirI[i]]]

    def updateFi(self):
    	# Update the the relaxation time constant
    	viscosity = self.mew/self.rho # 2d array type
    	tau = ((viscosity*6*self.dt/(self.dx)**2)+1)/2 ## also arraytype
    	# update Fi from FiStar using the collision operator
        conj_u = np.conj(self.u) 
        product = np.real(self.es * conj_u)
        self.s = self.ws * (3/self.c * product + 9/(2. * self.c ** 2) * product ** 2 - 3/(2 * self.c**2) * np.real(self.u * conj_u))
        self.FiEq = self.ws * self.rho + self.rho * self.s
        self.Fi = self.Fi - 1/tau * (self.FiStar - self.FiEq)

    def fullTimeStep(self):
        self.stream()
        self.inletOutlet()
        self.reflectOnMesh()
        self.updateRhoAndU()
        self.updateFi()

def plttt(latt):
    #X,Y = np.mgrid[0:.511:512j,0:.255:256j]
    X,Y = np.linspace(0,.511,512),np.linspace(0,.255,256)
    ux,uy =np.real(latt.u),np.imag(latt.u)
    fig = plt.figure()
    print np.shape(X),np.shape(Y),np.shape(ux),np.shape(uy)
    plt.streamplot(Y,X,uy,ux)
    plt.show()

class Visualization:
    
    def __init__(self,bLattice):
        self.lattice = bLattice
        
    def update(self,time):
        return
    
    def animate(self):
        return

