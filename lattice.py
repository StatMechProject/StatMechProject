import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


defaultX = 100
defaultY = 100
# Make the recutangular boundary mask
boundaryMask = np.zeros((defaultX,defaultY))
X,Y  = np.mgrid[0:defaultX,0:defaultY]
testMask =  (X < 53) * (X > 47) * (Y < 70) * (Y > 30)
boundaryMask[testMask] = 1
#circleMask = ((defaultX/2-X)**2 + (defaultY/2-Y)**2) < 25**2
#boundaryMask[circleMask] = 1

dividByZeroFudgeFactor = 1E-40


class Lattice:
    
    def __init__(self,reflectMesh=boundaryMask,Nx=defaultX,Ny=defaultY):
    	self.mu = .00001857 # constant for the mu
        self.Nx,self.Ny,self.Nvecs = Nx,Ny,9
        self.dx=.001 # 1mm spacing
        self.c = 340. # speed of sound 340 m/s
        self.dt = self.dx/self.c
        self.Fi = np.zeros((self.Nvecs,Nx,Ny)); 
        self.FiStar = self.Fi
        self.FiEq = self.Fi
        self.rho = np.zeros((Nx,Ny))
        self.u = np.zeros((Nx,Ny)) # 2d grid with complex numbers for vel vectors
        self.s = self.Fi
        self.es = np.array([0,1,1j,-1,-1j,1+1j,-1+1j,-1-1j,1-1j]).reshape((9,1,1)) #Our velocity basis
        self.ws = np.array([4/9.,1/9.,1/9.,1/9.,1/9.,1/36.,1/36.,1/36.,1/36.]).reshape((9,1,1)) #u weight
        self.reflectMesh = reflectMesh.astype(bool) # Mask containing borders an objects in our grid
        self.inletVelocity = 70. #m/s
        self.rho0 = 1.184 * self.dx**2 
        self.whereFluid = (boundaryMask==0) # All the places where we should be able to have fluid
        self.initFs()

        

    def initFs(self):
        # using density of air as 1.184 kg/m3 and our grid is 1m thick
        # Uniformly distribute over velocity and position space, remove fluid in boundaries

        self.Fi = np.ones((self.Nvecs,self.Nx,self.Ny)) * self.rho0 * self.whereFluid/9.
        es,c,Fi = self.es,self.c,self.Fi
        self.rho = np.sum(Fi,axis=0)
        sumEFi = np.sum(Fi*es,axis=0)
        self.u= c*sumEFi/(self.rho+dividByZeroFudgeFactor)*self.whereFluid

        
    def stream(self):
        es,Nvecs = self.es,self.Nvecs
        #Streaming step, move each of the distributions in their direction
        xShift,yShift = np.real(es).astype(int),np.imag(es).astype(int)
        self.FiStar = np.copy(self.Fi)
        for i in np.arange(Nvecs):
            if xShift[i]:
                self.FiStar[i] = np.roll(self.FiStar[i],xShift[i],axis=0)
            if yShift[i]:
                self.FiStar[i] = np.roll(self.FiStar[i],yShift[i],axis=1) 
    
    
    def reflectOnMesh(self):
        es,reflectMask,FiStar,Nvecs = self.es,self.reflectMesh,self.FiStar,self.Nvecs
        # Reflects at the 1's on the mesh

        revDirI = np.array([0,3,4,1,2,7,8,5,6])
        for i in [1,2,5,6]:
            swapCopy = np.copy(FiStar[i,reflectMask])
            FiStar[i,reflectMask] = FiStar[revDirI[i],reflectMask]
            FiStar[revDirI[i],reflectMask] = swapCopy
        
        # stream the reversed particles
        flippedFs = np.zeros((self.Nvecs,self.Nx,self.Ny))
        xShift,yShift = np.real(es).astype(int),np.imag(es).astype(int)
        flippedFs[:,reflectMask] = FiStar[:,reflectMask]
        for i in np.arange(Nvecs): 
            if xShift[i]:
                self.FiStar[i] += np.roll(flippedFs[i],xShift[i],axis=0)
            if yShift[i]:
                self.FiStar[i] += np.roll(flippedFs[i],yShift[i],axis=1) 
        self.FiStar[:,reflectMask] =0
        
    def updateRhoAndU(self):
        es,c,FiStar = self.es,self.c,self.FiStar
        self.rho = np.sum(FiStar,axis=0)
        sumEFi = np.sum(FiStar*es,axis=0)
        self.u= c*sumEFi/(self.rho+dividByZeroFudgeFactor) * self.whereFluid



    def vorticity(self):
        # single sided finite difference for del x u (curl u)
        ux,uy = np.real(self.u),np.imag(self.u)
        uxincy,uyincx = np.roll(ux,1,1),np.roll(uy,1,0)
        curlU = (uxincy-ux  -uyincx+uy)/self.dx
        return curlU



     def zouHe(self):
        FiStar = self.FiStar # Applies Zou Hue boundary conditions on top and bottom
        
        # inlet
        self.rho[0,1:-1] = (FiStar[0,0,1:-1] + FiStar[2,0,1:-1] + FiStar[4,0,1:-1] + 2 * (FiStar[6,0,1:-1] + FiStar[3,0,1:-1] + FiStar[7,0,1:-1])/(1+self.u0)
        FiStar[1,0,1:-1] = FiStar[3,0,1:-1] - (2/3) * self.rho[0,1:-1] * self.u0
        FiStar[7,0,1:-1] = FiStar[6,0,1:-1] + (1/2) * (FiStar[2,0,1:-1] - FiStar[4,0,1:-1]) - (1/6) * self.rho[0,1:-1] * self.u0
        FiStar[5,0,1:-1] = FiStar[7,0,1:-1] - (1/2) * (FiStar[2,0,1:-1] - FiStar[4,0,1:-1]) - (1/6) * self.rho[0,1:-1] * self.u0

        # outlet
        self.u[-1,1:-1] = -1 + (FiStar[0,-1,1:-1] + FiStar[4,-1,1:-1] FiStar[2,-1,1:-1] + 2 * (FiStar[1,-1,1:-1] + FiStar[8,-1,1:-1] + FiStar[5,-1,1:-1]))/self.rho0
        FiStar[3,-1,1:-1] = FiStar[1,-1,1:-1] - (2/3) * self.rho0 * self.u[-1,1:-1]
        FiStar[6,-1,1:-1] = FiStar[8,-1,1:-1] + (1/2) * (FiStar[4,-1,1:-1] - FiStar[2,-1,1:-1]) - (1/6) * self.rho0 * self.u[-1,1:-1]
        FiStar[7,-1,1:-1] = FiStar[5,-1,1:-1] - (1/2) * (FiStar[4,-1,1:-1] - FiStar[2,-1,1:-1]) - (1/6) * self.rho0 * self.u[-1,1:-1]

        #bottom 
        self.rho[:,-1] = FiStar[0,:,-1] + FiStar[1,:,-1] + FiStar[3,:,-1] + 2 * (FiStar[4,:,-1] + FiStar[7,:,-1] + FiStar[8,:,-1])
        FiStar[2,:,-1] = FiStar[4,:,-1]
        FiStar[5,:,-1] = FiStar[7,:,-1] - (1/2) * (FiStar[1,:,-1] - FiStar[3,:,-1]) + (1/2) * self.rho[:,-1] * self.u0
        FiStar[6,:,-1] = FiStar[8,:,-1] + (1/2) * (FiStar[1,:,-1] - FiStar[3,:,-1]) - (1/2) * self.rho[:,-1] * self.u0

        #top
        self.rho[:,0] = FiStar[0,:,0] + FiStar[1,:,0] + FiStar[3,:,0] + 2 * (FiStar[2,:,0] + FiStar[5,:,0] + FiStar[6,:,0])
        FiStar[4,:,0] = FiStar[2,:,0]
        FiStar[7,:,0] = FiStar[5,:,0] - (1/2) * (FiStar[3,:,0] - FiStar[1,:,0]) - (1/2) * self.rho[:,0] * self.u0
        FiStar[8,:,0] = FiStar[6,:,0] + (1/2) * (FiStar[3,:,0] - FiStar[1,:,0]) + (1/2) * self.rho[:,0] * self.u0


    def updateFi(self):
    	# Update the the relaxation time constant
    	viscosity = (self.dx**2)*self.mu/(self.rho+dividByZeroFudgeFactor) # 2d array type
    	tau = 2#((viscosity*6*self.dt/(self.dx)**2)+1)/2. ## also arraytype
        
    	# update Fi from FiStar using the collision operator
        conj_u = np.conj(self.u) 
        product = np.real(self.es * conj_u)
        self.s = self.ws * (3/self.c * product + 9/(2. * self.c ** 2) * product ** 2 - 3/(2 * self.c**2) * np.real(self.u * conj_u))
        self.FiEq = self.ws * self.rho + self.rho * self.s
        self.Fi = (self.Fi - (1/tau) * (self.FiStar - self.FiEq)) * self.whereFluid

    def fullTimeStep(self):
        self.stream() # Move distributions disregarding collision
        self.zouHe() 
        self.reflectOnMesh() # Reverse velocities of distributions that are out of bounds
        self.updateRhoAndU() # Calculate macroscopic quantities
        
        #self.updateFi() # Apply collision operator to determine proper Fi from FiEq

def plttt(latt):
    #X,Y = np.mgrid[0:.511:512j,0:.255:256j]
    X,Y = np.linspace(0,(self.dx * (defaultX - 1)),defaultX),np.linspace(0,(self.dx * (defaultY - 1)),defaultY)
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
