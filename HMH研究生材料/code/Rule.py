import pandas as pd
import random
from pandas.core.frame import DataFrame
import  numpy as np
random.seed(0)
redun = pd.read_csv("./benchmarks/FB15K/redun.txt",skiprows=[0],header=None,sep=" ")
none = pd.read_csv("./benchmarks/FB15K/none.txt",skiprows=[0],header=None,sep=" ")
rank = pd.read_csv("./result/FB15K/real/E.csv",sep=",",index_col=[0])
print(redun)
print(none)
redun.columns, none.columns = ["h","t","r"], ["h","t","r"]
print(rank)

maps={}
triple_both_n = []
triple_both_r = []
triple_both_ht = []
triple_both_h_or_t = []
# j=0
for i in range(none.shape[0]):
# for i in range(10):
    print(i)
    triple = tuple(none.iloc[i].values.tolist())
    both_ht = redun[((redun['h'] == none.iloc[i]['h']) |(redun['h'] == none.iloc[i]['t']))
                 & ((redun['t'] == none.iloc[i]['h']) |(redun['t'] == none.iloc[i]['t']))]
    if both_ht.values.tolist() != []:
        maps[triple] = both_ht.values.tolist()
        triple_both_ht.append(triple)
    else:
        both_h_or_t = redun[(redun['h'] == none.iloc[i]['h']) |(redun['h'] == none.iloc[i]['t'])
                 | (redun['t'] == none.iloc[i]['h']) |(redun['t'] == none.iloc[i]['t'])]
        if both_h_or_t.values.tolist() != []:
            maps[triple] = both_h_or_t.values.tolist()
            triple_both_h_or_t.append(triple)
        else:
            both_r = redun[redun['r'] == none.iloc[i]['r']]
            if both_r.values.tolist() != []:
                maps[triple] = both_r.values.tolist()
                triple_both_r.append(triple)
            else:
                maps[triple] = random.sample(redun.values.tolist(), 1)
                triple_both_n.append(triple)
raw_triple=[]
map_triple=[]
# random.seed(0)
# for i in maps.keys():
#     raw_triple.append(list(i))
#     map_triple.append(random.sample(maps[i], 1)[0])
k=0
for i in maps.keys():
    print(k)
    k+=1
    raw_triple.append(list(i))
    r = 20000
    for j in range(len(maps[i])):
        # print(maps[i][j])
        a = rank[(rank["h"]==maps[i][j][0]) & (rank["t"]==maps[i][j][1]) & (rank["r"]==maps[i][j][2])]
        # print(a)
        if (a['l_mr'].item() + a['r_mr'].item()) < r:
            r = (a['l_mr'].item() + a['r_mr'].item())
            m = a[["h","t","r"]].values.tolist()[0]
        if r ==2:
            break
    map_triple.append(m)

raw_triple=DataFrame(raw_triple)
raw_triple.columns = ["h","t","r"]
map_triple=DataFrame(map_triple)
map_triple.columns = ["map_h","map_t","map_r"]
data = pd.concat([raw_triple,map_triple],axis=1)
print(data)
data.to_csv("./benchmarks/FB15K/none_map.txt",sep=" ",index=None)