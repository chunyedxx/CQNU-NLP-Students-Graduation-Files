import pandas as pd
import re
import copy
import csv
f1 = open("../benchmarks/FB15K/train2id.txt","r")
f2 = open("../benchmarks/FB15K/test2id.txt","r")
f3 = open("../benchmarks/FB15K/valid2id.txt","r")
txt1 = f1.readlines()
txt2 = f2.readlines()
txt3 = f3.readlines()
print(txt1[:5])
a1 = []
# a2 = []
def TxtToList(txt):
    a = []
    for w in txt:
        w = w.replace("\n", "")
        w = w.split(" ", 2)
        a.append(w)
    del a[0]
    return a
alltri = []
allre = []
for txt in [txt1,txt2,txt3]:
    # TxtToList(txt)
    alltri+=TxtToList(txt)
# print(alltri)
print(len(alltri))
print(alltri[0])
reveralltri = copy.deepcopy(alltri)

for i in range(len(alltri)):
    allre.append(alltri[i][2])
    del alltri[i][2]
    del reveralltri[i][2]
    reveralltri[i][0], reveralltri[i][1] = reveralltri[i][1], reveralltri[i][0]



print("**************************************")
revers = []
# re = []
for i in range(len(alltri)):
# for i in range(5):
    print(i)
    # revers.append(alltri[i])
    # revers[-1].append(allre[i])
    if alltri[i] in reveralltri:
    # if alltri[i] in reveralltri and alltri[i][0] != alltri[i][1]:
        revers.append(alltri[i])
        revers[-1].append(allre[i])
print("revers",revers)
print(len(revers))


# with open('../analysis/redundata/FB15K/allrevers.txt', 'w', newline='') as csvfile:
#     writer  = csv.writer(csvfile)
#     writer.writerow(['h','t','r'])
#     for row in revers:
#         writer.writerow(row)
