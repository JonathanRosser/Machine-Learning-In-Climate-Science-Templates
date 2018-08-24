import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#Create an ensemble, define the first M terms of the vector as the parameters and the following N terms of the vector as the data ( G(u) )
M=2
N=2
#ztrue contains the true parameters and the observed data (y) calculated from them which we are using to train the model.
ztrue=np.zeros((N+M))
#Define the true parameters
#for i in range(0,M):
#    ztrue[i]=np.random.uniform()
ztrue[0]=1 #h 
ztrue[1]=10 #F
#ztrue[2]=10 #c
#ztrue[3]=10 #b
K=36 #Number of slow variables 
J=10 #Number of fast variables per slow variable
longspin=1000
fastspin=100
longtime=10000
fasttime=1000

#Define functions for use in the Linear Model
def LModel(z):
    G=np.zeros(N)
    Mat=np.zeros((N,M))
    Mat[0,0]=1
    Mat[0,1]=2
    Mat[1,0]=3
    Mat[1,1]=4
    G=np.matmul(Mat,z)+np.random.normal(loc=0.0,scale=0.0001,size=N)
    return G

#Conduct an initial long run to determine the true data
print("Calculate True data")
GLONG=np.zeros((100,N))
for i in range(0,100):
    GLONG[i,:]=LModel(ztrue[:M])
ztrue[M:]=np.mean(GLONG,axis=0)
print("Calculate Covariance")
r=0.0000001
Gamma=(r**2)*np.diag((np.var(GLONG[:,0]),np.var(GLONG[:,1])))



#Output the data
Q=400 #Number of ensemble members
ITER=4 #Number of iterations
parameters=np.zeros((M,Q*ITER))
Output=np.zeros((N,Q*ITER))

#Prediction step

#Create an initial ensemble, Q is the number of ensemble members
z=np.zeros((N+M,Q))
for i in range(0,M):
    if i == 0:
        for j in range(0,Q):
            print(j)
            z[i,j]=np.random.normal(loc=0.0,scale=1.0)
    if i ==1:
        for j in range(0,Q):
            print(j)
            z[i,j]=np.random.normal(loc=10.0,scale=10.0)
    if i ==2:
        for j in range(0,Q):
            print(j)
            z[i,j]=np.exp(np.random.normal(loc=2.0,scale=0.1))
    if i==3:
        for j in range(0,Q):
            print(j)
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
                print(count,i)



            #Sample mean
            zbar=np.mean(z,axis=1)
            print("zbar= ",zbar)
            if zbar[M]==0 and zbar[M+1]==0 and zbar[M+2]==0:
                print("Solver has failed")
                break
            #Sample covariance
            #Mean of outer products
            MOP=np.zeros((N+M,N+M))
            for i in range(0,Q):
                MOP=MOP+np.outer(z[:,i],np.transpose(z[:,i]))/Q
            #Covariance
            C=MOP-np.outer(zbar,np.transpose(zbar))






            #Analysis step
            print("Analysis")
            #Define the Kalman gain

            H=np.append(np.zeros((N,M)),np.identity(N),axis=1)
            A=np.matmul(C,np.transpose(H))
            B=np.matmul(H,np.matmul(C,np.transpose(H)))+Gamma
            KAL=np.transpose(np.linalg.solve(np.transpose(B),np.transpose(A)))


            #Generate random data

            yrand=np.zeros((N,Q))
            for i in range(0,Q):
                yrand[:,i]=ztrue[M:M+N]+np.random.normal(loc=0.0,scale=np.sqrt(np.diag(Gamma)))


            #Update the ensemble members
            z=z-np.matmul(KAL,np.matmul(H,z))+np.matmul(KAL,yrand)







            #Convergence
            #Compute the mean of the parameter update:
            u=np.mean(z,axis=1)[0:M]
            print("u=",u)

            #Check for convergence:
            tau=1
            #SHOULD THIS BE -1 or -0.5 for the power of the matrix
            #alpha=np.linalg.norm(np.matmul(np.linalg.matrix_power(Gamma,-1)[M:N+M,M:N+M],ztrue[M:N+M]-np.matmul(G,u)))
            #beta=tau*np.linalg.norm(np.matmul(np.linalg.matrix_power(Gamma,-1)[M:N+M,M:N+M],etadag))
            #if alpha <= beta:
            #    break
            count += 1

#np.save("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4.npy",parameters)
#np.save("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4.npy", Output)



