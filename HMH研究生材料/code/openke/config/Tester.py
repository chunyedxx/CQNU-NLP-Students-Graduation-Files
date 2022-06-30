# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
import csv
import pandas as pd
import math
import torch.nn.functional as F





class Tester(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True, path = None, path2 = None):
        base_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../release/Base.so"))
        self.path = path
        self.path2 = path2
        self.lib = ctypes.cdll.LoadLibrary(base_file)
        self.lib.testHead.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.testTail.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
        self.lib.test_link_prediction.argtypes = [ctypes.c_int64]

        self.lib.getTestLinkMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkHit1.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkFMRR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkFMR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkFHit10.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkFHit3.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkFHit1.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkL_rank.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkL_filter_rank.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkR_rank.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkR_filter_rank.argtypes = [ctypes.c_int64]
        self.lib.testHeadH.argtypes = [ctypes.c_int64]
        self.lib.testHeadT.argtypes = [ctypes.c_int64]
        self.lib.testHeadR.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkL_mrr.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkR_mrr.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkL_fmrr.argtypes = [ctypes.c_int64]
        self.lib.getTestLinkR_fmrr.argtypes = [ctypes.c_int64]




        self.lib.getTestLinkMRR.restype = ctypes.c_float
        self.lib.getTestLinkMR.restype = ctypes.c_float
        self.lib.getTestLinkHit10.restype = ctypes.c_float
        self.lib.getTestLinkHit3.restype = ctypes.c_float
        self.lib.getTestLinkHit1.restype = ctypes.c_float
        self.lib.getTestLinkFMRR.restype = ctypes.c_float
        self.lib.getTestLinkFMR.restype = ctypes.c_float
        self.lib.getTestLinkFHit10.restype = ctypes.c_float
        self.lib.getTestLinkFHit3.restype = ctypes.c_float
        self.lib.getTestLinkFHit1.restype = ctypes.c_float
        self.lib.getTestLinkL_rank.restype = ctypes.c_float
        self.lib.getTestLinkL_filter_rank.restype = ctypes.c_float
        self.lib.getTestLinkR_rank.restype = ctypes.c_float
        self.lib.getTestLinkR_filter_rank.restype = ctypes.c_float
        self.lib.testHeadH.restype = ctypes.c_float
        self.lib.testHeadT.restype = ctypes.c_float
        self.lib.testHeadR.restype = ctypes.c_float
        self.lib.getTestLinkL_mrr.restype = ctypes.c_float
        self.lib.getTestLinkR_mrr.restype = ctypes.c_float
        self.lib.getTestLinkL_fmrr.restype = ctypes.c_float
        self.lib.getTestLinkR_fmrr.restype = ctypes.c_float


        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'mode': data['mode']
        })
    #
    # def get_entemb(self):
    #
    #     for i in range(self.reent_tot):
    #         j = int(list(self.new_redunmap.keys())[i])
    #         self.ent_embeddings.weight.data[j] = self.reent_embeddings.weight.data[i]
    #     # self.redunent_embedding.weight.data[i] = torch.from_numpy(np.array(redunentity_dic[i]))
    #     for i in range(self.noent_tot):
    #         j = int(list(self.new_nonemap.keys())[i])
    #         self.ent_embeddings.weight.data[j] = self.noent_embeddings.weight.data[i]

    # self.noneent_embedding.weight.data[i] = torch.from_numpy(np.array(noneentity_dic[i]))

    def run_link_prediction(self, type_constrain = False):
        import time
        self.lib.initTest()
        self.data_loader.set_sampling_mode('link')
        # data = self.Data(self.path)
        if type_constrain:
            type_constrain = 1
        else:
            type_constrain = 0
        training_range = tqdm(self.data_loader)
        # print("8888888888888888888888888888888")
        result = []
        # sco = {'hsco':[],'tsco':[]}
        # self.model.get_entemb()
        for index, [data_head, data_tail] in enumerate(training_range):
            # print("index", index)
            # print("[data_head, data_tail]",[data_head, data_tail])
            start = time.time()
            lscore = self.test_one_step(data_head)
            # sco['hsco'].append(lscore)
            # print("lscore",lscore,len(lscore))
            rscore = self.test_one_step(data_tail)
            # sco['tsco'].append(rscore)
            # print("rscore", rscore, len(rscore))
            self.lib.testHead(lscore.__array_interface__["data"][0], index, type_constrain)
            self.lib.testTail(rscore.__array_interface__["data"][0], index, type_constrain)
            l_rank = self.lib.getTestLinkL_rank(type_constrain)
            l_filter_rank = self.lib.getTestLinkL_filter_rank(type_constrain)
            r_rank = self.lib.getTestLinkR_rank(type_constrain)
            r_filter_rank = self.lib.getTestLinkR_filter_rank(type_constrain)
            l_mrr = self.lib.getTestLinkL_mrr(type_constrain)
            r_mrr = self.lib.getTestLinkR_mrr(type_constrain)
            l_fmrr = self.lib.getTestLinkL_fmrr(type_constrain)
            r_fmrr = self.lib.getTestLinkR_fmrr(type_constrain)
            testhead_h = self.lib.testHeadH(index)
            testhead_t = self.lib.testHeadT(index)
            testhead_r = self.lib.testHeadR(index)
            result.append([int(testhead_h), int(testhead_t), int(testhead_r), int(l_rank), int(r_rank),int(l_filter_rank),\
                           int(r_filter_rank),float(l_mrr),float(r_mrr),float(l_fmrr),float(r_fmrr)])
            end = time.time()
            t = end-start
            # print("Index %d | time: %f " % (index, t))
        # print(sco)
        # testh = pd.DataFrame(sco['hsco'])
        # testh.to_csv('/home/llv19/PycharmProjects/HMH/OpenKE/Score/F15K/testHcsv10.csv', encoding='gbk')
        # testt = pd.DataFrame(sco['hsco'])
        # testt.to_csv('/home/llv19/PycharmProjects/HMH/OpenKE/Score/F15K/testTcsv10.csv', encoding='gbk')
        df = pd.DataFrame(result, columns=['h','t','r','l_mr','r_mr','l_fmr','r_fmr','l_mrr','r_mrr','l_fmrr','r_fmrr'])
        # print(df.loc[:,'r_mr'])
        df.to_csv(self.path2, sep=',')
        l_mr = df['l_mr'].sum() / df.shape[0]
        r_mr = df['r_mr'].sum() / df.shape[0]
        l_fmr = df['l_fmr'].sum() / df.shape[0]
        r_fmr = df['r_fmr'].sum() / df.shape[0]
        mr = (l_mr+ r_mr)/ 2
        fmr = (l_fmr+ r_fmr)/ 2
        # print(l_mr, r_mr, l_fmr, r_fmr, mr, fmr)

        l_mrr = df['l_mrr'].sum() / df.shape[0]
        r_mrr = df['r_mrr'].sum() / df.shape[0]
        l_fmrr = df['l_fmrr'].sum() / df.shape[0]
        r_fmrr = df['r_fmrr'].sum() / df.shape[0]
        mrr = (l_mrr + r_mrr) / 2
        fmrr = (l_fmrr + r_fmrr) / 2
        # print(l_mrr, r_mrr, l_fmrr, r_fmrr, mrr, fmrr)

        l_hit10 = df[df['l_mr'] <= 10].shape[0] / df.shape[0]
        r_hit10 = df[df['r_mr'] <= 10].shape[0] / df.shape[0]
        l_fhit10 = df[df['l_fmr'] <= 10].shape[0] / df.shape[0]
        r_fhit10 = df[df['r_fmr'] <= 10].shape[0] / df.shape[0]
        hit10 = (l_hit10+ r_hit10)/ 2
        fhit10 = (l_fhit10+ r_fhit10)/ 2
        # print(l_hit10, r_hit10, l_fhit10, r_fhit10, hit10, fhit10)
        test = {"l_mr": [format(l_mr, '.5f')], "r_mr": [format(r_mr, '.5f')], "l_fmr": [format(l_fmr, '.5f')], \
                     "r_fmr": [format(r_fmr, '.5f')], "mr": [format(mr, '.5f')], "fmr": [format(fmr, '.5f')], \
                     "l_mrr": [format(l_mrr, '.5f')], "r_mrr": [format(r_mrr, '.5f')], "l_fmrr": [format(l_fmrr, '.5f')], \
                     "r_fmrr": [format(r_fmrr, '.5f')], "mrr": [format(mrr, '.5f')], "fmrr":[format(fmrr, '.5f')], \
                "l_hit10": [format(l_hit10, '.5f')], "r_hit10": [format(r_hit10, '.5f')], "l_fhit10": [format(l_fhit10, '.5f')], \
                "r_fhit10": [format(r_fhit10, '.5f')], "hit10": [format(hit10, '.5f')], "fhit10": [format(fhit10, '.5f')]
                }
        print("mr",test["mr"],"fmr",test["fmr"],"hit10",test["hit10"],"fhit10",test["fhit10"],)

        # test = pd.DataFrame(test)
        # print(test)
        # test.to_csv(self.path2, sep=',')
        return test["fhit10"]



        self.lib.test_link_prediction(type_constrain)
        # r_mr,l_mr,r_fmr,l_fmr = 0,0,0,0
        # r_mrr = 0
        # for i in range(len(testtri)):
        #     r_mr += testtri[i][4]
        #     l_mr += testtri[i][3]
        #     r_fmr += testtri[i][6]
        #     l_fmr += testtri[i][5]
        #     r_mrr += testtri[i][8]
        # print(r_mr/len(testtri), r_mrr/len(testtri),l_mr/len(testtri),r_fmr/len(testtri),l_fmr/len(testtri))
        # # mrr = dict(mrr = self.lib.getTestLinkMRR(type_constrain))
        # mr = dict(mr = self.lib.getTestLinkMR(type_constrain))
        # hit10 = dict(hit10 = self.lib.getTestLinkHit10(type_constrain))
        # # hit3 = dict(hit3 = self.lib.getTestLinkHit3(type_constrain))
        # # hit1 = dict(hit1 = self.lib.getTestLinkHit1(type_constrain))
        # fmrr = dict(fmrr = self.lib.getTestLinkFMRR(type_constrain))
        # fmr = dict(fmr = self.lib.getTestLinkFMR(type_constrain))
        # fhit10 = dict(fhit10 = self.lib.getTestLinkFHit10(type_constrain))
        # # fhit3 = dict(fhit3 = self.lib.get     TestLinkFHit3(type_constrain))
        # # fhit1 = dict(fhit1 = self.lib.getTestLinkFHit1(type_constrain))
        #header = ['h', 't', 'r', 'l_rank', "r_rank", "l_frank", "r_frank", 'l_mrr', 'r_mrr', 'l_fmrr', 'r_fmrr', ]

        # with open(self.path, "w", encoding='utf8',newline='') as f:
        #     writer = csv.writer(f,delimiter=',')
        #     for row in result:
        #         writer.writerow(row)


    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        res_mx = 0.0
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx

    def run_triple_classification(self, threshlod = None):
        self.lib.initTest()
        self.data_loader.set_sampling_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, [pos_ins, neg_ins] in enumerate(training_range):
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)

            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_pos))]
            score.append(res_neg)

        score = np.concatenate(score, axis = -1)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0

        return acc, threshlod

