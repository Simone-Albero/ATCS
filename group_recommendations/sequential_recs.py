import progressbar as pb
import pandas as pd
import numpy as np
import os
from group_based_cf import getSatisfaction

def dfSplit(df, part_num = 3):
    num_rows = len(df)
    part_size = num_rows // 3

    return np.split(df, [part_size, part_size*2])

def overallSatisfaction(user, group_rec):
    u_sats = map(getSatisfaction, group_rec) # Parametri incompatibili!

    return np.mean(u_sats)

def groupSatisfaction(u_sats):
    pass

def main():
    df_path = os.path.join(os.getcwd(), 'group_recommendations', 'dataset', 'ratings.csv')
    df = pd.read_csv(df_path)
    
    ITER_NUM = 3
    iter1, iter2, iter3 = dfSplit(df, ITER_NUM)





    np.random.seed(42)
    users = np.random.choice(df['userId'].unique(), 3, replace=False)

if __name__ == "__main__":
    main()