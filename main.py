import UDEA
import argparse
import misc
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(description='UDEA Parameters')
    parser.add_argument('--convTol',type=float,required=False, default=1e-8)
    parser.add_argument('--maxUncrty',type=float,required=False, default=0)
    parser.add_argument('--delta',type=float,required=False, default=0.1)
    parser.add_argument('--alpha',type=float,required=False, default=0.4)
    parser.add_argument('--eps',type=float,required=False, default=1e-8)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    X,Y = misc.process_dataset('period1.csv')
    out = UDEA.UDEA_parallelized(X,Y,args.convTol,args.maxUncrty,args.delta,args.alpha,args.eps)
    df_out = pd.DataFrame(columns=["efficiency", "uncertainity", "capability", "exit_flag", "noinal_eff"])
    for x in range(len(out)):
        dic = out[x]
        for i in range(len(dic)):
            df_out.loc[list(dic.keys())[i]] = list(dic.values())[i]
    df_out.index += 1
    df_out = df_out.sort_index(ascending=True)
    df_out.to_csv("out.csv")
    print("Success!")
