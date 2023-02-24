from rsome import ro
from rsome import norm
import numpy as np
import pandas as pd
from rsome import grb_solver as grb
import gurobipy



def getABC(X,Y,i):
    X_new = np.array(X).T #N X D
    D = len(X_new[0])
    Y_new = np.array(Y).T# M X D
    y_i = Y_new[:,[i]] # in paper i from 1 to n and we won't follow it
    x_i = X_new[:,[i]] # in paper i from 1 to n and we won't follow it
    up = np.concatenate((-Y_new,y_i, np.zeros((len(Y_new),1))),axis=1)
    low = np.concatenate((X_new,np.zeros((len(X_new),1)),-x_i),axis=1)
    A = np.concatenate((up,low), axis = 0)
    B = np.concatenate((np.ones((1,D)),np.zeros((1,1)), np.zeros((1,1))), axis = 1)
    B = np.concatenate((B, np.zeros((1, D + 2))), axis = 0)
    B[1,-2] = 1
    C = np.zeros((D+2,1))
    C[-1,0] = 1
    return A, B, C

def get_robust_efficiency(X,Y,i,sigma = None):
    A,B,C = getABC(X,Y,i)
    '''
    if (not sigma):
        sigma = [0.5]*len(A[0])
    '''
    model = ro.Model()
    neta = model.dvar((len(X)+2))
    z = [model.rvar((len(X))) for i in range(len(X[0]) + len(Y[0])) ]
    z_set0 = (norm(z[i],2) <= 1 for i in range(len(X[0]) + len(Y[0])) )
    model.min((C.T)[0]@neta)
    model.st(B[0]@ neta == 1,B[1] @ neta == 1 )
    model.st(neta >= (np.zeros(len(X)+2)))
    for x in range(len(A)):
        A_x = A[x]
        I = np.eye(len(A[0]))
        if (x < len(Y[0])):
            mat = np.zeros((len(X), len(A[0])), int)
            np.fill_diagonal(mat,1)
            mat[x,-2] = -1
            model.st(((A_x + sigma[x]*(z[x])@mat)@neta <= 0).forall(z_set0))
        else:
            mat = np.zeros((len(X), len(A[0])), int)
            np.fill_diagonal(mat,1)
            mat[x -len(Y[0]),-1] = -1
            model.st(((A_x + sigma[x]*(z[x])@mat)@neta <= 0).forall(z_set0))
        #model.st(((A_x)@neta <= 0))
    try:
        model.solve(grb,  display = False)
    except gurobipy.GurobiError:
        return -1
    try :
        eff = (neta.get())[-1]
    except RuntimeError:
        return -1
    return eff

def DEA_eff(X,Y):
    eff_arr = []
    D = len(X)
    for i in range(D):
        A,B,C = getABC(X,Y,i)
        '''
        if (not sigma):
            sigma = [0.5]*len(A[0])
        '''
        model = ro.Model()
        neta = model.dvar((len(X)+2))
        #z = [model.rvar((len(X))) for i in range(len(X[0]) + len(Y[0])) ]
        #z_set0 = (norm(z[i],2) <= 1 for i in range(len(X[0]) + len(Y[0])) )
        model.min((C.T)[0]@neta)
        model.st(B[0]@ neta == 1,B[1] @ neta == 1 )
        model.st(neta >= (np.zeros(len(X)+2)))
        for x in range(len(A)):
            A_x = A[x]
            I = np.eye(len(A[0]))
            model.st(((A_x)@neta <= 0))
            #model.st(((A_x)@neta <= 0))
        try:
            model.solve(grb,  display = False)
        except gurobipy.GurobiError:
            return -1
        try :
            eff = (neta.get())[-1]
        except RuntimeError:
            return -1
        eff_arr.append(eff)
    return eff_arr
    