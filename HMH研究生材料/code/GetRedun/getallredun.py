R = [4, 1029, 1028, 8, 1032, 526, 1038, 1041, 535, 1051, 540, 1062, 1064, 553, 1068, 561, 1076, 55,\
     569, 570, 59, 1083, 574, 65, 584, 81, 593, 83, 85, 597, 598, 1109, 1116, 606, 1120, 99, 611, 1128,\
     624, 1137, 1139, 118, 630, 121, 1146, 639, 642, 1159, 1163, 141, 658, 149, 151, 665, 1178, 1180, 671,\
     678, 1190, 680, 171, 173, 685, 1199, 688, 177, 1202, 1212, 1215, 704, 193, 705, 709, 1225, 204, 1231,\
     720, 211, 213, 725, 727, 728, 1241, 1248, 738, 228, 741, 742, 743, 233, 1257, 747, 1258, 749, 240, 753,\
     244, 245, 759, 1271, 1276, 255, 259, 773, 262, 776, 1290, 1294, 1295, 272, 1296, 1298, 1299, 1301, 1302,\
     1303, 280, 284, 798, 1313, 291, 293, 1318, 296, 811, 300, 305, 308, 310, 1342, 1343, 837, 838, 327, 332,\
     852, 341, 344, 345, 347, 860, 353, 354, 872, 362, 876, 879, 888, 383, 384, 895, 900, 393, 395, 400, 406,\
     919, 409, 412, 932, 437, 955, 450, 967, 970, 972, 466, 979, 983, 985, 477, 480, 483, 997, 999, 1001, \
     1005, 496, 1008, 499, 500, 1011, 502, 503] #184

import pandas as pd
import numpy as np
traindata = pd.read_table("../benchmarks/FB15K/train2id.txt",\
                   header=None,sep=' ',error_bad_lines=False,encoding='utf-8',skiprows=[0])
testdata = pd.read_table("../benchmarks/FB15K/test2id.txt",\
                   header=None,sep=' ',error_bad_lines=False,encoding='utf-8',skiprows=[0])
validdata = pd.read_table("../benchmarks/FB15K/valid2id.txt",\
                   header=None,sep=' ',error_bad_lines=False,encoding='utf-8',skiprows=[0])
traindata.columns, testdata.columns, validdata.columns = ['h', 't', 'r'], ['h', 't', 'r'], ['h', 't', 'r']
# traindata.columns= ['h', 't', 'r']

# data["0"].value_counts()
redun = []
redun = pd.DataFrame(columns = ["h", "t", "r"])
print(redun)
for i in R:
    print(i)
    traintri = traindata.loc[traindata['r'].isin([i])]
    testtri = testdata.loc[testdata['r'].isin([i])]
    validtri = validdata.loc[validdata['r'].isin([i])]
    # print("testtri",testtri)
    redun = redun.append(traintri)
    redun = redun.append(testtri)
    redun = redun.append(validtri)
    # print("testredun",testredun)
print(redun)
# redun.to_csv('./analysis/redundata/FB15K/allredun.csv',index=None)




