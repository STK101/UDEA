import numpy as np
import pandas as pd
import cvxpy as cp
import dill as pickle

def process_dataset(path, ds = 0):
    X = None
    Y = None
    if (ds == 0):
        df = pd.read_csv(path, index_col= 0)
        df['Annual Return'] = df['Annual Return'].apply(lambda x : float(x[:-1]))
        df['Total Risk'] = df['Total Risk'].apply(lambda x : float(x[:-1]))
        df['Abs. Win Rate'] = df['Abs. Win Rate'].apply(lambda x : 100 - float(x[:-1]))
        if(min(df['Annual Return']) < 0):
            df['Annual Return'] -= min(df['Annual Return'])
        X = (df[['Total Risk','Abs. Win Rate']]).values.tolist()
        Y = (df[['Annual Return']]).values.tolist()
    elif (ds == 1):
        df = pd.read_csv(path, index_col= 0)
        X = (df[["IP1", "IP2"]]).values.tolist()
        Y = (df[["OP1", "OP2"]]).values.tolist()
    return X,Y

def forward_difference(func, sigma, eps = 1e-4,maxUncrty = 1):
    n = len(sigma)
    if (sum(np.array(sigma) < 0)):
        return [0.]*n
    del_func = np.array([0.]*n)
    for i in range(n):
        del_x = [0.]*n
        del_x[i] = eps
        fact = 0
        sigma_new = np.clip(np.array(sigma) + np.array(del_x), 0, maxUncrty)
        itr = 0
        while (np.linalg.norm(sigma_new) >= maxUncrty and itr < 1e2):
            fact += 1
            itr += 1
            sigma_new = np.clip(np.array(sigma) + pow((1/2),fact)*np.array(del_x), 0, maxUncrty)
        if(itr == 1e2):
            del_func[i] = 0
        else:
            del_func[i] = (-func(sigma) + func(sigma_new.tolist()))/(eps* pow((1/2),fact))
    return del_func

def sigma_enhancer(sigma, h, der_m,env):
    #der_m is column too
    n = len(h)# h is column matrix
    if (np.linalg.norm(der_m) < 1e-8 or np.linalg.norm(h) < 1e-8):
        return [0.]*n
    x = cp.Variable(n)
    soc_constraints = [cp.SOC(1, x), h.T @ x == 0 ]
    prob = cp.Problem(cp.Minimize(der_m.T@x),soc_constraints)
    prob.solve(solver=cp.GUROBI,env= env)#, TimeLimit = 2)
    return x.value

def bisection01(d,func,eps,sigma,Tol, maxUncrty = 1):  
    lb = 0
    ub = 1
    b = (lb+ub)/2
    d = np.array(d)
    sigma = np.array(sigma)
    itr = 0
    #print("bisection Enter")
    returnable = False
    while(True and itr < 1e2):
        sigma_new = np.clip(sigma + b*d, 0, maxUncrty)
        itr += 1
        if (np.linalg.norm(sigma_new) >= maxUncrty):
            ub = b
            nb = (lb + ub)/2
            returnable = False
        elif (func(sigma) <= func(sigma_new) +eps):
            returnable = True
            lb = b
            nb = (lb + ub)/2
        else:
            returnable = True
            ub = b
            nb = (lb + ub)/2
        if(abs(b - nb) <= Tol):
            #print("bisection Exit")
            return b
        else:
            b = nb
    #print("bisection Exit")
    if (returnable):
        return b
    return 0

