import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random
import time
runtime=np.zeros(10)
for SAMPLE in range(0,10):
    start=time.time()

    #Create an ensemble, define the first M terms of the vector as the parameters and the following N terms of the vector as the data ( G(u) )
    M=4
    N=100
    #ztrue contains the true parameters and the observed data (y) calculated from them which we are using to train the model.
    ztrue=np.zeros((N+M))
    #Define the true parameters
    #for i in range(0,M):
    #    ztrue[i]=np.random.uniform()
    ztrue[0]=1 #h 
    ztrue[1]=10 #F
    ztrue[2]=10 #c
    ztrue[3]=10 #b
    K=36 #Number of slow variables 
    J=10 #Number of fast variables per slow variable
    longspin=1000
    fastspin=100
    longtime=10000
    fasttime=1000
    hstep=1
    delta=0.1
    Mat=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENFlow/Linear/Matrix.npy")

    #Define functions for use in the Linear Model
    def LModel(z):
        G=np.zeros(N)
        G=np.matmul(Mat,z)
        return G





    def INNER(j,k):
        a=z[M:M+N,k]-zbar[M:]
        #print(a)
        b=yrand[:,j,k]-z[M:M+N,j]
        #print(b)
        a2=np.matmul(a,Gammainv)
        d=np.inner(a2,b)/Q
        return d,a2,b


    Gamma=np.identity(2)

    #Conduct an initial long run to determine the true data
    #Conduct an initial long run to determine the true data
    #print("Calculate True data")
    GLONG=np.zeros((100,N))
    GLONG[:,:]=LModel(ztrue[:M])
    ztrue[M:]=np.mean(GLONG,axis=0)
    #print("Calculate Covariance")
    r=0.001
    #Gamma=(r**2)*np.diag((np.var(GLONG[:,0]),np.var(GLONG[:,1])))
    Gamma=np.identity(2)
    Gammainv=np.linalg.inv(Gamma)
    #Gammainv=np.diag(1.0/np.diag(Gamma))
    #Output the data
    Q=400 #Number of ensemble members
    ITER=100 #Number of iterations
    parameters=np.zeros((M,Q*ITER))
    Output=np.zeros((N,Q*ITER))

    #Prediction step
    #np.random.seed(2)
    #Create an initial ensemble, Q is the number of ensemble members
    z=np.zeros((N+M,Q))
    for i in range(0,M):
        if i == 0:
            for j in range(0,Q):
                #print(j)
                z[i,j]=np.random.normal(loc=0.0,scale=1.0)
        if i ==1:
            for j in range(0,Q):
                #print(j)
                z[i,j]=np.random.normal(loc=10.0,scale=10.0)
        if i ==2:
            for j in range(0,Q):
                #print(j)
                z[i,j]=np.exp(np.random.normal(loc=2.0,scale=0.1))
        if i==3:
            for j in range(0,Q):
                #print(j)
                z[i,j]=np.random.normal(loc=5,scale=10.0)



    t=np.arange(0.0,fasttime/100,0.01)

    count=0

    while count<ITER:
                #Prediction step

                #Calculate the new ensemble predictions

                for i in range(0,Q):
                    z[M:M+N,i]=LModel(tuple(z[0:M,i]))
                    #Update the output data
                    parameters[:,count*Q+i]=z[:M,i]
                    Output[:,count*Q+i]=z[M:M+N,i]
                    #print(count,i)



                #Sample mean
                zbar=np.mean(z,axis=1)
                #print("zbar= ",zbar)
                if zbar[M]==0 and zbar[M+1]==0 and zbar[M+2]==0:
                    print("Solver has failed")
                    break




                #Analysis step
                #print("Analysis")
                yrand=np.zeros((Q,Q,N))
                #yrand[:,:,:]=ztrue[M:M+N]+np.random.normal(loc=0.0,scale=np.sqrt(np.diag(Gamma)),size=(Q,Q,N))
                yrand[:,:,:]=ztrue[M:M+N]
                ztilde=np.transpose(np.transpose(z)-zbar)
                #CppSOP=np.zeros((N,N))
                #for i in range(0,Q):
                #    CppSOP=CppSOP+np.outer(z[M:,i],np.transpose(ztilde[M:,i]))
                #Cpp=np.tensordot(z[M:,:],ztilde[M:,:],axes=(1,1))/Q
                #Cpp=CppSOP/Q

                #Gammasum=Gamma+Cpp
                #Gammainv=np.linalg.inv(Gammasum)


                #Define the D matrix
                #D=np.zeros((Q,Q))
                #A=np.zeros((Q,Q,2))
                #B=np.zeros((Q,Q,2))
                #for j in range(0,Q):
                #    for k in range(0,Q):
                #        D[j,k],A[j,k,:],B[j,k,:]=INNER(j,k)
                #print(np.linalg.norm(D))
                A2=np.transpose(z[M:,:])-zbar[M:]
                B2=yrand[0,:,:,]-np.transpose(z[M:,:])
                D2=np.transpose(np.tensordot(np.matmul(A2,Gammainv),B2,axes=(1,1)))/Q




                #Update the ensemble members
                step=hstep/(np.linalg.norm(D2)+delta)
                #step=0.001
                #print("step= {}".format(step))
                #for j in range(0,M):
                #    z[j,:]=z[j,:]+float(step)*np.matmul(D,z[j,:])
                z=z+float(step)*np.transpose(np.matmul(D2,np.transpose(z)))






                #Convergence
                #Compute the mean of the parameter update:
                u=np.mean(z,axis=1)[0:M]
                print("u=",u)

                #Check for convergence:
                #tau=1
                #SHOULD THIS BE -1 or -0.5 for the power of the matrix
                #alpha=np.linalg.norm(np.matmul(np.linalg.matrix_power(Gamma,-1)[M:N+M,M:N+M],ztrue[M:N+M]-np.matmul(G,u)))
                #beta=tau*np.linalg.norm(np.matmul(np.linalg.matrix_power(Gamma,-1)[M:N+M,M:N+M],etadag))
                #if alpha <= beta:
                #    break
                count += 1
    end=time.time()
    runtime[SAMPLE]=end-start
    np.save("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENFlow/Linear/ENFlowparameters400x4sample{}.npy".format(SAMPLE),parameters)
    np.save("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENFlow/Linear/ENFlowOutput400x4sample{}.npy".format(SAMPLE), Output)
np.save("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENFlow/Linear/runtime.npy",runtime)


