import pandas as pd


DCR = "../analysis/redundata/FB15K/testDCR.csv"
redun1 = "../analysis/redundata/FB15K/allredun.csv"
reverst = "../analysis/redundata/FB15K/allrevers.txt"
data = "../benchmarks/FB15K/test2id.txt"
DCR = pd.read_table(DCR, header=0, sep=',', error_bad_lines=False, encoding='utf-8')
redun1 = pd.read_table(redun1, header=0, sep=',', error_bad_lines=False, encoding='utf-8')
reverst = pd.read_table(reverst, header=0, sep=',', error_bad_lines=False, encoding='utf-8')
train = pd.read_table(data, header=None, sep=' ',error_bad_lines=False, encoding='utf-8', skiprows=[0])
train.columns = ['h','t','r']


# re = reverst
re = pd.concat([redun1, reverst, DCR])
re = re.drop_duplicates(keep="first",inplace=False)      #近似冗余数据加上反向关系
# print(re)
datare=pd.merge(re, train, on=['h', 't', 'r'], how='inner')
print(data)
datanone = pd.concat([data, train])
trainnone = datanone.drop_duplicates(keep=False, inplace=False)
print(trainnone)
datare.to_csv('../analysis/redundata/FB15K/testredun.txt')
trainnone.to_csv('../analysis/redundata/FB15K/testnone.txt')

