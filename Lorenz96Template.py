import numpy as np
from scipy.integrate import odeint

#These variables are required for the Lorenz model to run

longspin=100
longtime=1000
t=np.arange(0.0,longtime/100,0.01)
x0=np.random.rand(K*(J+1))





def Lorenz96(x,t,h,F,c,b):
    #Create vector for the derivatives
    d=np.zeros(K*(J+1))
    #Slow variable case
    #Consider the edge cases first (i=1,2,K):
    d[0]=-x[K-1]*(x[K-2]-x[1])-x[0]+F-h*c*np.mean(x[K:K+J])
    d[1]=-x[0]*(x[K-1]-x[2])-x[1]+F-h*c*np.mean(x[K+J:K+2*J])
    d[K-1]=-x[K-2]*(x[K-3]-x[0])-x[K-1]+F-h*c*np.mean(x[K+J*(K-1):K+J*(K)])
    #General case:
    for i in range(2,K-1):
        d[i]=-x[i-1]*(x[i-2]-x[i+1])-x[i]+F-h*c*np.mean(x[K+J*i:K+J*(i+1)])

#Fast variable case
    for l in range(0,K):
        #Consider the edge cases first (i=1,J-1,J):
        N=K+l*J
        d[N]=c*(-b*x[N+1]*(x[N+2]-x[N+J-1])-x[N]+h*x[l]/J)
        d[N+J-1]=c*(-b*x[N]*(x[N+1]-x[N+J-2])-x[N+J-1]+h*x[l]/J)
        d[N+J-2]=c*(-b*x[N+J-1]*(x[N]-x[N+J-3])-x[N+J-2]+h*x[l]/J)
    #General case:
        for i in range(1,J-2):
            N=K+l*J+i
            d[N]=c*(-b*x[N+1]*(x[N+2]-x[N-1])-x[N]+h*x[l]/J)
    return d


#Running average function
def runavg(func,x,W):
    #Function is the function which will be calculated over each of the windows, i.e. a mean or variance etc
    #x is the array over which the function will be calculated
    #W is the window length
    print(func)
    if func==np.cov:
        y=np.zeros((len(x)-W,K*(J+1),K*(J+1)))
        for i in range(0,len(x)-W):
            y[i,:,:]=np.cov(x[i:W+i],rowvar=False)
        return y
    else:
        y=np.zeros(len(x)-W)
        for i in range(0,len(x)-W):
            y[i]=func(x[i:W+i])
        return y

#Function to calculate the XY term which will allow for correlation between X and Y which might be significant
def XY(x,time,spin):
    y=np.zeros((time-spin,K))
    for i in range(0,K):
        y[:,i]=np.mean(x[spin:,K+i*J:K+(i+1)*J],axis=1)
    XY=np.mean(np.multiply(x[spin:,:K],y),axis=1)
    return XY

def XY2(x,time,spin):
    y=np.zeros((time-spin,K))
    for i in range(0,K):
        y[:,i]=np.mean(x[spin:,K+i*J:K+(i+1)*J],axis=1)
    XY=np.mean(np.multiply(x[spin:,:K],y))
    return XY

def output(x,time,spin):
    y=np.zeros(N)
    y[0]=np.mean(x[spin:,:K]) #Mean of the slow variables
    y[1]=np.mean(x[spin:,K:]) #Mean of the fast variables
    y[2]=np.mean(np.square(x[spin:,:K])) #Mean of the squared slow variables
    y[3]=np.mean(np.square(x[spin:,K:])) #Mean of the squared fast variables
    y[4]=XY2(x,time,spin) #Mean of XY
    print(y)
    return y


ztrue=np.zeros((N+M))
#Define the true parameters
#for i in range(0,M):
#    ztrue[i]=np.random.uniform()
ztrue[0]=1 #h 
ztrue[1]=10 #F
ztrue[2]=10 #c
ztrue[3]=10 #b



#To get output from the Lorenz model:

ztrue[M:M+N]=output(odeint(Lorenz96,x0,t,tuple(ztrue[0:M])),longtime,longspin)






