import pandas as pd
import numpy as np

data = pd.read_table("/home/llv19/PycharmProjects/HMH/OpenKE/benchmarks/FB15K/train2id.txt",\
                          header=None,sep=' ',error_bad_lines=False,encoding='utf-8',skiprows=[0])
data.columns = ['h', 't', 'r']
R=[307,947,529,338,451,482,513,174,385]
DCR = pd.DataFrame(columns = ["h", "t", "r"])
for i in R:
    print(i)
    traintri = data.loc[data['r'].isin([i])]
    # print("testtri",testtri)
    DCR = DCR.append(traintri)
    # print("testredun",testredun)
print(DCR)
DCR.to_csv('../analysis/redundata/FB15K/trainDCR.csv',index=None)