import numpy as np
import pandas as pd
#import matlab.engine
import misc
import RobustDEA
#from tqdm import tqdm as tqdm

import multiprocess as multiprocessing
from multiprocess import Process
import dill as pickle

def m(sigma, R_set):
    pass

def UDEA_parallelized(X,Y,convTol, maxUncrty, delta, alpha, eps, env):
    # we use 6 parralel processes
    D = len(X)
    '''
    eng = matlab.engine.start_matlab()
    X = matlab.double(X)
    Y = matlab.double(Y)
    io_vrs_1 = eng.dea(X,Y, 'orient', 'io', 'rts', 'vrs')
    eng.quit()
    '''

    noinal_efficiency = np.asarray(RobustDEA.DEA_eff(X,Y,env))
    #print(noinal_efficiency)
    #np.savetxt('DEA_effs.txt', noinal_efficiency)
    a1 = (np.arange(0,D,6)).tolist()
    a2 = (np.arange(1,D,6)).tolist()
    a3 = (np.arange(2,D,6)).tolist()
    a4 = (np.arange(3,D,6)).tolist()
    a5 = (np.arange(4,D,6)).tolist()
    a6 = (np.arange(5,D,6)).tolist()
    return_dict = (multiprocessing.Manager()).dict()
    #func_temp = lambda ids,jn,return_dict : UDEA(X,Y,convTol, maxUncrty, delta, alpha, eps,ids,jn,return_dict)
    p1 = Process(target=UDEA, args=(X,Y,convTol, maxUncrty, delta, alpha, eps,a1,1,return_dict,noinal_efficiency,env))
    p2 = Process(target=UDEA, args=(X,Y,convTol, maxUncrty, delta, alpha, eps,a2,2,return_dict,noinal_efficiency,env))
    p3 = Process(target=UDEA, args=(X,Y,convTol, maxUncrty, delta, alpha, eps,a3,3,return_dict,noinal_efficiency,env))
    p4 = Process(target=UDEA, args=(X,Y,convTol, maxUncrty, delta, alpha, eps,a4,4,return_dict,noinal_efficiency,env))
    p5 = Process(target=UDEA, args=(X,Y,convTol, maxUncrty, delta, alpha, eps,a5,5,return_dict,noinal_efficiency,env))
    p6 = Process(target=UDEA, args=(X,Y,convTol, maxUncrty, delta, alpha, eps,a6,6,return_dict,noinal_efficiency,env))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    return return_dict.values()

def UDEA(X,Y,convTol, maxUncrty, delta, alpha, eps,ids,job_num, return_dict,noinal_efficiency,env):
    D = len(X) # D X N
    N = len(X[0])
    M = len(Y[0])
    '''
    eng = matlab.engine.start_matlab()
    X = matlab.double(X)
    Y = matlab.double(Y)
    io_vrs_1 = eng.dea(X,Y, 'orient', 'io', 'rts', 'vrs')
    eng.quit()
    noinal_efficiency = np.asarray(io_vrs_1['eff'])
    '''

    Ude_eff_uncrty_capability = dict()#[[0.0,0.0,0.0,0.0]]*D #efficiency, uncertainity, capability {0 : incapable, 1: weakly capable, 2 : capable }, exit_flag(either sigma new > 1 ), noinal_eff,  
    #8 cores so 8 parallelizationx
    for i in ids: #range(D)
        print("Started - ", i+1)
        if (i<0):#i == 46 or i == 57 or i == 31 and  maxUncrty == 1):#i == 7 or i == 22 or i == 32 or i == 47): [fuckers still fucking up can't do this shit no more will have to debug]
            print("Done - ", i+1)
            continue
        if(noinal_efficiency[i] >= 1):
            Ude_eff_uncrty_capability[i] = [1.,0.,0.,0, noinal_efficiency[i]]
        else:
            eff_func = lambda s : RobustDEA.get_robust_efficiency(X,Y,i,s,env)
            searchFlag = True
            sigma = [0.]*(M + N)
            beta = 0.
            d = [0.]*(M + N)
            exitflag = 0
            itr_cap = 0
            while(searchFlag and itr_cap < 1e2):
                #print("Loop entered", i+1)
                #sigma = [0.]*(M + N)
                #print("F Dif 1 ", i+1)
                h = misc.forward_difference(eff_func,sigma, delta,maxUncrty)
                sigma_new =(np.array(sigma) + alpha*np.array(h)).tolist()
                #if (sum(np.array(sigma_new) < 0) or sum(np.array(sigma_new) > maxUncrty) ):
                    #beta = 0
                sigma_new = np.clip(sigma_new, 0, maxUncrty).tolist()
                '''
                if sum(np.array(sigma_new) < 0):
                    sigma_new[sigma_new < 0] = 0
                    exitflag += 1
                else:
                    exitflag += 2
                break
                '''
                sigma = sigma_new
                #print("F Dif 2 ", i+1)
                h = misc.forward_difference(eff_func,sigma, delta,maxUncrty)
                der_m = misc.forward_difference(lambda x : np.linalg.norm(x),sigma, 1e-3,maxUncrty)
                #print("sigma enhancer entered")
                d = misc.sigma_enhancer(sigma,h,der_m,env)
                #print("sigma enhancer exited")
                beta = misc.bisection01(d,eff_func,eps,sigma,1e-4,maxUncrty)
                if(beta == 0):
                    print("Beta 0 For DMU - ", i+1)
                #if (sum(np.array(sigma) + beta*np.array(d) < 0) or sum(np.array(sigma) + beta*np.array(d)> maxUncrty) ):
                    #beta = 0
                sigma_new = (np.clip(np.array(sigma) + beta*np.array(d), 0, maxUncrty)).tolist()
                '''
                if sum(np.array(sigma) + beta*np.array(d) < 0):
                    exitflag += 1
                else:
                    exitflag += 2
                
                break
                '''
                if(np.linalg.norm(sigma_new) > maxUncrty or abs(eff_func(sigma) - eff_func(sigma_new)) < convTol ):
                    if(np.linalg.norm(sigma_new) > maxUncrty):
                        exitflag += np.linalg.norm(sigma_new)
                        sigma = ((sigma_new)/(np.linalg.norm(sigma_new)))*maxUncrty
                    elif (abs(eff_func(sigma) - eff_func(sigma_new)) < convTol):
                        exitflag += eff_func(sigma)
                    searchFlag = False
                else:
                    ucrty_gain = np.linalg.norm(sigma_new) - np.linalg.norm(sigma)
                    eff_gain =  eff_func(sigma_new) - eff_func(sigma)
                    sigma = (sigma_new)
                    if(eff_func(sigma_new) >= 1):
                        searchFlag = False

                itr_cap += 1
            if (itr_cap == 1e2):
                exitflag = -1
            final_eff = eff_func(sigma)
            final_uncrty = np.linalg.norm(sigma)
            if (final_eff >= 1 and final_uncrty <= maxUncrty):
                Ude_eff_uncrty_capability[i] = [final_eff, final_uncrty , 0,exitflag,noinal_efficiency[i]]
            elif (final_eff >= 1 and final_uncrty > maxUncrty):
                Ude_eff_uncrty_capability[i] = [final_eff, final_uncrty , 1,exitflag,noinal_efficiency[i]]
            else:
                Ude_eff_uncrty_capability[i] = [final_eff, final_uncrty , 2, exitflag,noinal_efficiency[i]]
                
        print("Done - ", i+1)
    return_dict[job_num] = Ude_eff_uncrty_capability
    return Ude_eff_uncrty_capability