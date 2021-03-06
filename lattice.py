import numpy as np

defaultX = 1200
defaultY = 600
# Make the recutangular boundary mask
boundaryMask = np.zeros((defaultX,defaultY))
X,Y  = np.mgrid[0:defaultX,0:defaultY]
#testMask =  (X < 53) * (X > 47) * (Y < 70) * (Y > 30)
#boundaryMask[testMask] = 1
#circleMask = ((defaultX * 4./9.-X)**2 + (defaultY/2.-Y)**2) < 15.**2
#boundaryMask[circleMask] = 1
plateMask = (Y==defaultY/2.) * (X > defaultX * 1./20.) * (X < defaultX * 19./20.)
boundaryMask[plateMask] = 1

divideByZeroFudgeFactor = 1E-25


class Lattice:
    
    def __init__(self,reflectMesh=boundaryMask,Nx=defaultX,Ny=defaultY):
        self.mu = .00001857 #*100 # *1000# constant for the mu
        self.Nx,self.Ny,self.Nvecs = Nx,Ny,9
        self.dx=.000001 # 1mm spacing
        self.c = 340. # speed of sound 340 m/s
        self.dt = self.dx/self.c
        self.rho = np.zeros((Nx,Ny))
        self.u = np.zeros((Nx,Ny)) # 2d grid with complex numbers for vel vectors
        self.es = np.array([0,1,1j,-1,-1j,1+1j,-1+1j,-1-1j,1-1j]).reshape((9,1,1)) #Our velocity basis
        self.ws = np.array([4/9.,1/9.,1/9.,1/9.,1/9.,1/36.,1/36.,1/36.,1/36.]).reshape((9,1,1)) #u weight
        self.reflectMesh = reflectMesh.astype(bool) # Mask containing borders an objects in our grid
        self.u0 = 70./self.c #m/s
        self.rho0 = 1.184 * self.dx**2 
        self.whereFluid = (boundaryMask==0).astype(bool) # All the places where we should be able to have fluid
        self.initFs()

        

    def initFs(self):
        # using density of air as 1.184 kg/m3 and our grid is 1m thick
        # Uniformly distribute over velocity and position space, remove fluid in boundaries

        self.Fi = (np.ones((self.Nvecs,self.Nx,self.Ny)) + 0.2 * (np.random.rand(self.Nvecs,self.Nx,self.Ny) -1./2.) )* self.rho0 * self.whereFluid/9.       
        self.FiStar = self.Fi
        self.FiEq = self.Fi
        self.s = self.Fi
        
        es,c,Fi = self.es,self.c,self.Fi
        self.rho = np.sum(Fi,axis=0)
        sumEFi = np.sum(Fi*es,axis=0)
        self.u= c*sumEFi/(np.maximum(self.rho,divideByZeroFudgeFactor)) * self.whereFluid

        
    def stream(self):
        es,Nvecs = self.es,self.Nvecs
        #Streaming step, move each of the distributions in their direction
        xShift,yShift = np.real(es).astype(int),np.imag(es).astype(int)
        
        for i in np.arange(Nvecs):
            if xShift[i]!=0:
                #print(self.FiStar[i])
                self.FiStar[i,:,:] = np.roll(self.FiStar[i,:,:],xShift[i,:,:],axis=0)
                #if xShift[i]==1:  self.FiStar[i,0,:]=0
                #if xShift[i]==-1: self.FiStar[i,-1,:]=0
                #print(self.FiStar[i])
            if yShift[i]!=0:
                self.FiStar[i,:,:] = np.roll(self.FiStar[i,:,:],yShift[i,:,:],axis=1)
                #if yShift[i]==1:  self.FiStar[i,:,0]=0
                #if yShift[i]==-1: self.FiStar[i,:,-1]=0


    def bounceBack(self):
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
        self.u= c*sumEFi/(np.maximum(self.rho,divideByZeroFudgeFactor)) * self.whereFluid




    def vorticity(self):
        # single sided finite difference for del x u (curl u)
        ux,uy = np.real(self.u),np.imag(self.u)
        uxincy,uyincx = np.roll(ux,1,1),np.roll(uy,1,0)
        curlU = (uxincy-ux  -uyincx+uy)/self.dx
        return curlU



    def zouHe(self):

        FiStar = self.FiStar # Applies Zou Hue boundary conditions on top and bottom
        #FiStar[1,0,:] = self.rho0
        #FiStar[3,-1,:] = self.rho0
        # inlet
        trho = (FiStar[0,0,1:-1] + FiStar[2,0,1:-1] + FiStar[4,0,1:-1] + 2. * (FiStar[6,0,1:-1] + FiStar[3,0,1:-1] + FiStar[7,0,1:-1])) / (1. - self.u0)
        
        FiStar[1,0,1:-1] = FiStar[3,0,1:-1] + (2./3.) * trho * self.u0
        FiStar[5,0,1:-1] = FiStar[7,0,1:-1] - (1./2.) * (FiStar[2,0,1:-1] - FiStar[4,0,1:-1]) + (1./6.) * trho * self.u0
        FiStar[8,0,1:-1] = FiStar[6,0,1:-1] + (1./2.) * (FiStar[2,0,1:-1] - FiStar[4,0,1:-1]) + (1./6.) * trho * self.u0

        # outlet
        tu = -1 + (FiStar[0,-1,1:-1] + FiStar[4,-1,1:-1] + FiStar[2,-1,1:-1] + 2. * (FiStar[1,-1,1:-1] + FiStar[8,-1,1:-1] + FiStar[5,-1,1:-1])) / self.rho0
        FiStar[3,-1,1:-1] = FiStar[1,-1,1:-1] - (2./3.) * self.rho0 * tu
        FiStar[6,-1,1:-1] = FiStar[8,-1,1:-1] + (1./2.) * (FiStar[4,-1,1:-1] - FiStar[2,-1,1:-1]) - (1./6.) * self.rho0 * tu
        FiStar[7,-1,1:-1] = FiStar[5,-1,1:-1] - (1./2.) * (FiStar[4,-1,1:-1] - FiStar[2,-1,1:-1]) - (1./6.) * self.rho0 * tu

        #bottom 
        trho = FiStar[0,:,-1] + FiStar[1,:,-1] + FiStar[3,:,-1] + 2. * (FiStar[4,:,-1] + FiStar[7,:,-1] + FiStar[8,:,-1])
        FiStar[2,:,-1] = FiStar[4,:,-1]
        FiStar[5,:,-1] = FiStar[7,:,-1] - (1./2.) * (FiStar[1,:,-1] - FiStar[3,:,-1]) + (1./2.) * trho * self.u0
        FiStar[6,:,-1] = FiStar[8,:,-1] + (1./2.) * (FiStar[1,:,-1] - FiStar[3,:,-1]) - (1./2.) * trho * self.u0

        #top
        trho = FiStar[0,:,0] + FiStar[1,:,0] + FiStar[3,:,0] + 2. * (FiStar[2,:,0] + FiStar[5,:,0] + FiStar[6,:,0])
        FiStar[4,:,0] = FiStar[2,:,0]
        FiStar[7,:,0] = FiStar[5,:,0] - (1./2.) * (FiStar[3,:,0] - FiStar[1,:,0]) - (1./2.) * trho * self.u0
        FiStar[8,:,0] = FiStar[6,:,0] + (1./2.) * (FiStar[3,:,0] - FiStar[1,:,0]) + (1./2.) * trho * self.u0



    def updateFi(self):
        # Update the the relaxation time constant
        viscosity = self.mu/(np.maximum(self.rho,divideByZeroFudgeFactor)) * (self.dx**2) # 2d array type
        tau = ((viscosity*6.*self.dt/(self.dx**2))+1.)/2. ## also arraytype
        
        # update Fi from FiStar using the collision operator
        product = np.real(self.u)*np.real(self.es)+np.imag(self.u)*np.imag(self.es)

        self.s = self.ws * (3. * product / self.c  + 9. / (2. * self.c ** 2) * product ** 2 - 3. / (2. * self.c**2) * np.abs(self.u)**2)


        self.FiEq = self.ws * self.rho + self.rho * self.s
        #print "before stream", np.min(self.FiStar)
        self.Fi = self.FiStar - (1/tau) * (self.FiStar - self.FiEq) * self.whereFluid
        

    def fullTimeStep(self):

        self.FiStar = np.copy(self.Fi)
        self.stream() # Move distributions disregarding collision
        self.zouHe() # Adjust inlet and outlet distrubutions
        self.bounceBack() # Reverse velocities of distributions that are out of bounds
        self.updateRhoAndU() # Calculate macroscopic quantities
        self.updateFi() # Apply collision operator to determine proper Fi from FiEq
         

    def convergence(self):
        
        u1 = np.copy(self.u)
        rho1 = np.copy(self.rho)
        self.fullTimeStep()
        u2 = np.copy(self.u)
        rho2 = np.copy(self.rho)
        
        uConv = np.max(np.abs(u1 - u2))
        rhoConv = np.max(np.abs(rho1 - rho2))
        
        return {"u":uConv, "rho":rhoConv}
    
