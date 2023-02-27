#from rsome import ro
#from rsome import norm
import numpy as np
import pandas as pd
import cvxpy as cp
#from rsome import grb_solver as grb
import gurobipy
import dill as pickle



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

def get_robust_efficiency(X,Y,i,sigma ,env  ):
    A,B,C = getABC(X,Y,i)
    neta =  cp.Variable((len(X)+2))
    soc_constraints = []
    for x in range(len(A)):
        A_x = A[x]
        I = np.eye(len(A[0]))
        if (x < len(Y[0])):
            mat = np.zeros((len(X), len(A[0])))
            np.fill_diagonal(mat,1)
            mat[x,-2] = -1
            soc_constraints.append(cp.SOC((-A_x.T @ neta), (sigma[x] * mat @ neta)))
            #model.st(((A_x + sigma[x]*(z[x])@mat)@neta <= 0).forall(z_set0))
        else:
            mat = np.zeros((len(X), len(A[0])))
            np.fill_diagonal(mat,1)
            mat[x -len(Y[0]),-1] = -1
            soc_constraints.append(cp.SOC((-A_x.T @ neta), (sigma[x] *mat @ neta)))
            #model.st(((A_x + sigma[x]*(z[x])@mat)@neta <= 0).forall(z_set0))
    zeros =  (np.zeros(len(X)+2))
    neg_I =  -(np.eye((len(X)+2))).astype(float)
    prob = cp.Problem(cp.Minimize(((C.T)[0]).T@neta),
                    soc_constraints + [B[0].T@ neta == 1, B[1].T @ neta == 1, neg_I@neta <= zeros] )
    try :
        prob.solve(solver=cp.GUROBI , env = env)
    except cp.error.SolverError:
        return 1
    try :
        eff = (neta.value)[-1]
    except TypeError:
        eff = 1
    if (eff > 1):
        return 1
    return eff
    '''
    if (not sigma):
        sigma = [0.5]*len(A[0])
    '''
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
'''
def DEA_eff(X,Y,env):
    eff_arr = []
    D = len(X)
    for i in range(D):
        A,B,C = getABC(X,Y,i)
        neta =  cp.Variable((len(X)+2))
        constraints = []
        for x in range(len(A)):
            A_x = A[x]
            constraints.append(A_x.T @ neta <= 0)

        zeros =  (np.zeros(len(X)+2))
        neg_I =  -(np.eye((len(X)+2))).astype(float)
        prob = cp.Problem(cp.Minimize(((C.T)[0]).T@neta),
                        constraints + [B[0].T@ neta == 1, B[1].T @ neta == 1, neg_I@neta <= zeros] )
        try :
            prob.solve(solver=cp.GUROBI , env = env)
        except cp.error.SolverError:
            return 1
        try :
            eff = (neta.value)[-1]
        except TypeError:
            eff = 1
        if (eff > 1):
            return 1
        return eff
    return eff_arr
    