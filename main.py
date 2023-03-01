import UDEA
import argparse
import misc
import pandas as pd
import gurobipy
import dill as pickle

def parse_args():
    parser = argparse.ArgumentParser(description='UDEA Parameters')
    parser.add_argument('--dataset',type=int,required=False, default=0)
    parser.add_argument('--convTol',type=float,required=False, default=1e-8)
    parser.add_argument('--maxUncrty',type=float,required=False, default=0)
    parser.add_argument('--delta',type=float,required=False, default=0.1)
    parser.add_argument('--alpha',type=float,required=False, default=0.4)
    parser.add_argument('--eps',type=float,required=False, default=1e-8)
    parser.add_argument('--colab', type = int, required = False, default = 0)
    parser.add_argument('--jobs', type = int, required = False, default = 6)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if (args.colab > 0):
        params = {
        "WLSACCESSID": '2ab307d3-acc2-4903-8eaf-0a021bd86360',
        "WLSSECRET": '8f130215-c8d1-4312-ab50-cf63c33a0121',
        "LICENSEID": 937219,
        }
        env = gurobipy.Env(params=params)
    else:
        env = gurobipy.Env() 
    if (args.dataset == 0 ):
        X,Y = misc.process_dataset('period1.csv',args.dataset)
    else:
        X,Y = misc.process_dataset('toy.csv',args.dataset)
    out = UDEA.UDEA_parallelized(X,Y,args.convTol,args.maxUncrty,args.delta,args.alpha,args.eps,env,args.jobs)
    df_out = pd.DataFrame(columns=["efficiency", "uncertainity", "capability", "exit_flag", "noinal_eff"])
    for x in range(len(out)):
        dic = list(out)[x]
        for i in range(len(dic)):
            df_out.loc[list(dic.keys())[i]] = list(dic.values())[i]
    df_out.index += 1
    df_out = df_out.sort_index(ascending=True)
    df_out.to_csv("out.csv")
    print("Success!")
