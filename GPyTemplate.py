import GPy
import numpy as np
from GPy.models.gp_regression import GPRegression as reg

#This is a program which will make a Gaussian Process and store the Gaussian Process in a dictionary called Models and then delete the Gaussian Process from the other memory. This is very important as the GPy package seems to struggle over-writing data. The parameter M is the number of input parameters I think.

M=4 #Number of input parameters


def gpmake(X,Y,name):
    print("Making {}".format(name))
    kernel=GPy.kern.RBF(M)
    noise=0.1
    normalizer=False
    gp=reg(X,Y,kernel,noise_var=noise,normalizer=normalizer)
    gp.optimize()
    Models[name]=gp
    del gp

