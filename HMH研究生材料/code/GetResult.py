import pandas as pd
data = pd.read_csv("./result/FB15K/new/H_DEM11.csv",index_col=[0])
reduntest = pd.read_csv("./benchmarks/FB15K/testredun.txt",sep=" ",skiprows=[0],header=None)
nonetest = pd.read_csv("./benchmarks/FB15K/testnone.txt",sep=" ",skiprows=[0],header=None)
reduntest.columns,nonetest.columns=["h","t","r"],["h","t","r"]


# print(data)
# print(reduntest)
# print(nonetest)
re_rank = pd.merge(data,reduntest)
no_rank = pd.merge(data,nonetest)
print(re_rank.shape)
print(no_rank.shape)
re_rmr = re_rank["r_mr"].mean()
re_lmr = re_rank["l_mr"].mean()
re_mr = (re_rmr+re_lmr)/2
re_rmrr = re_rank["r_mrr"].mean()
re_lmrr = re_rank["l_mrr"].mean()
re_mrr = (re_rmrr+re_lmrr)/2
re_rhit10 = (re_rank[re_rank["r_mr"]<=10])["r_mr"].count()/re_rank.shape[0]
re_lhit10 = (re_rank[re_rank["l_mr"]<=10])["l_mr"].count()/re_rank.shape[0]
re_hit10 = (re_rhit10+re_lhit10)/2
print("redun")
print(re_mr,re_mrr,re_hit10)

re_rfmr = re_rank["r_fmr"].mean()  #157.65870752483738
re_lfmr = re_rank["l_fmr"].mean()
re_fmr = (re_rfmr+re_lfmr)/2
re_rfmrr = re_rank["r_fmrr"].mean()
re_lfmrr = re_rank["l_fmrr"].mean()
re_fmrr = (re_rfmrr+re_lfmrr)/2
re_rfhit10 = (re_rank[re_rank["r_fmr"]<=10])["r_fmr"].count()/re_rank.shape[0]
re_lfhit10 = (re_rank[re_rank["l_fmr"]<=10])["l_fmr"].count()/re_rank.shape[0]
re_fhit10 = (re_rfhit10+re_lfhit10)/2
print(re_fmr,re_fmrr,re_fhit10)

no_rmr = no_rank["r_mr"].mean()
no_lmr = no_rank["l_mr"].mean()
no_mr = (no_rmr+no_lmr)/2
no_rmrr = no_rank["r_mrr"].mean()
no_lmrr = no_rank["l_mrr"].mean()
no_mrr = (no_rmrr+no_lmrr)/2
no_rhit10 = (no_rank[no_rank["r_mr"]<=10])["r_mr"].count()/no_rank.shape[0]
no_lhit10 = (no_rank[no_rank["l_mr"]<=10])["l_mr"].count()/no_rank.shape[0]
no_hit10 = (no_rhit10+no_lhit10)/2
print("none")
print(no_mr,no_mrr,no_hit10)

no_rfmr = no_rank["r_fmr"].mean()
no_lfmr = no_rank["l_fmr"].mean()
no_fmr = (no_rfmr+no_lfmr)/2
no_rfmrr = no_rank["r_fmrr"].mean()
no_lfmrr = no_rank["l_fmrr"].mean()
no_fmrr = (no_rfmrr+no_lfmrr)/2
no_rfhit10 = (no_rank[no_rank["r_fmr"]<=10])["r_fmr"].count()/no_rank.shape[0]
no_lfhit10 = (no_rank[no_rank["l_fmr"]<=10])["l_fmr"].count()/no_rank.shape[0]
no_fhit10 = (no_rfhit10+no_lfhit10)/2
print(no_fmr,no_fmrr,no_fhit10)


no_rmr = no_rank["r_mr"].mean()
no_lmr = no_rank["l_mr"].mean()
no_mr = (no_rmr+no_lmr)/2
no_rmrr = no_rank["r_mrr"].mean()
no_lmrr = no_rank["l_mrr"].mean()
no_mrr = (no_rmrr+no_lmrr)/2

mr = (no_rank["r_mr"].sum()+no_rank["l_mr"].sum()+re_rank["r_mr"].sum()+re_rank["l_mr"].sum())/(no_rank.shape[0]+re_rank.shape[0])/2
mrr = (no_rank["r_mrr"].sum()+no_rank["l_mrr"].sum()+re_rank["r_mrr"].sum()+re_rank["l_mrr"].sum())/(no_rank.shape[0]+re_rank.shape[0])/2
hit10 = ((no_rank[no_rank["l_mr"]<=10])["l_mr"].count() + (no_rank[no_rank["r_mr"]<=10])["r_mr"].count() +\
        (re_rank[re_rank["l_mr"]<=10])["l_mr"].count()+(re_rank[re_rank["r_mr"]<=10])["r_mr"].count())/(no_rank.shape[0]+re_rank.shape[0])/2
print("all")
print(mr,mrr,hit10)

fmr = (no_rank["r_fmr"].sum()+no_rank["l_fmr"].sum()+re_rank["r_fmr"].sum()+re_rank["l_fmr"].sum())/(no_rank.shape[0]+re_rank.shape[0])/2
fmrr = (no_rank["r_fmrr"].sum()+no_rank["l_fmrr"].sum()+re_rank["r_fmrr"].sum()+re_rank["l_fmrr"].sum())/(no_rank.shape[0]+re_rank.shape[0])/2
fhit10 = ((no_rank[no_rank["l_fmr"]<=10])["l_fmr"].count() + (no_rank[no_rank["r_fmr"]<=10])["r_fmr"].count() +\
        (re_rank[re_rank["l_fmr"]<=10])["l_fmr"].count()+(re_rank[re_rank["r_fmr"]<=10])["r_fmr"].count())/(no_rank.shape[0]+re_rank.shape[0])/2
print(fmr,fmrr,fhit10)

mr = (data["r_mr"].sum()+data["l_mr"].sum())/(data.shape[0])/2
mrr = (data["r_mrr"].sum()+data["l_mrr"].sum())/(data.shape[0])/2
hit10 = ((data[data["l_mr"]<=10])["l_mr"].count() + (data[data["r_mr"]<=10])["r_mr"].count())/(data.shape[0])/2
print("all")
print(mr,mrr,hit10)

fmr = (data["r_fmr"].sum()+data["l_fmr"].sum())/(data.shape[0])/2
fmrr = (data["r_fmrr"].sum()+data["l_fmrr"].sum())/(data.shape[0])/2
fhit10 = ((data[data["l_fmr"]<=10])["l_fmr"].count() + (data[data["r_fmr"]<=10])["r_fmr"].count())/(data.shape[0])/2
print(fmr,fmrr,fhit10)


