import codecs
import pandas as pd



def RedunData(file1):
    redundata = pd.read_csv(file1,skiprows=[0],sep=" ",header = None)
    redundata.columns = ["h","t","r"]
    r_h = redundata["h"].tolist()
    r_t = redundata["t"].tolist()
    r_r = redundata["r"].tolist()
    return r_h,r_t,r_r
