import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import matplotlib.pyplot as plt
# import numpy
import pandas as pd
from torch.autograd import Variable
class TransE_DEM(Model):

    def __init__(self, ent_tot, rel_tot=0, dim=100, p_norm=1, norm_flag=True, margin=None, epsilon=None, Lue_num=300,Lur_num=300,none_map=None):
        super(TransE_DEM, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.Lue_num = Lue_num
        self.Lur_num = Lur_num
        self.none_map = none_map
        self.softmax = nn.Softmax(dim=1)
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.Lue_embeddings = nn.Embedding(self.Lue_num, self.dim)
        self.Lur_embeddings = nn.Embedding(self.Lur_num, self.dim)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.Lue_embeddings.weight.data)
            nn.init.xavier_uniform_(self.Lur_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        # print("max(batch_h)",max(batch_h))
        # print("max(batch_t)",max(batch_t))
        # print("max(batch_r)",max(batch_r))
        # print('batch_h', batch_h)
        # print('batch_t', batch_t)
        # print('batch_r', batch_r)

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        sim_h = torch.mm(h, self.Lue_embeddings.weight.t())
        att_h = self.softmax(sim_h)
        att_head = torch.mm(att_h, self.Lue_embeddings.weight)/2+h/2

        sim_t = torch.mm(t, self.Lue_embeddings.weight.t())
        att_t = self.softmax(sim_t)
        att_tail = torch.mm(att_t, self.Lue_embeddings.weight)/2+t/2

        sim_r = torch.mm(r, self.Lur_embeddings.weight.t())
        att_r = self.softmax(sim_r)
        att_rel = torch.mm(att_r, self.Lur_embeddings.weight)/2+r/2
        score = self._calc(att_head, att_tail, att_rel, mode)
        # score = self._calc(att_head, att_tail, att_rel, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def Regularization(self,data,batch_size):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        raw_triple = torch.cat([batch_h[:batch_size].unsqueeze(1),batch_t[:batch_size].unsqueeze(1),batch_r[:batch_size].unsqueeze(1)],dim=1)
        raw_triple = raw_triple.cpu().numpy()
        raw_triple = pd.DataFrame(raw_triple, columns=["h", "t","r"])
        # print(raw_triple)
        map_triple = pd.merge(raw_triple, self.none_map)
        map_triple = map_triple[['map_h','map_t','map_r']]
        # print(map_triple)

        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        sim_h = torch.mm(h, self.Lue_embeddings.weight.t())
        att_h = self.softmax(sim_h)
        att_head = torch.mm(att_h, self.Lue_embeddings.weight)/2+h/2

        sim_t = torch.mm(t, self.Lue_embeddings.weight.t())
        att_t = self.softmax(sim_t)
        att_tail = torch.mm(att_t, self.Lue_embeddings.weight)/2+t/2

        sim_r = torch.mm(r, self.Lur_embeddings.weight.t())
        att_r = self.softmax(sim_r)
        att_rel = torch.mm(att_r, self.Lur_embeddings.weight)/2+r/2

        triple = att_head[:batch_size,:] + att_rel[:batch_size,:] - att_tail[:batch_size,:]
        # print(triple)
        # triple = triple.unsqueeze(dim=-2)

        r_h = map_triple["map_h"].tolist()
        r_t = map_triple["map_t"].tolist()
        r_r = map_triple["map_r"].tolist()
        r_h = Variable(torch.tensor(r_h, dtype=torch.long).cuda())
        r_t = Variable(torch.tensor(r_t, dtype=torch.long).cuda())
        r_r = Variable(torch.tensor(r_r, dtype=torch.long).cuda())
        r_head = self.ent_embeddings(r_h)
        r_tail = self.ent_embeddings(r_t)
        r_rel = self.rel_embeddings(r_r)
        r_sim_h = torch.mm(r_head, self.Lue_embeddings.weight.t())
        r_att_h = self.softmax(r_sim_h)
        r_att_head = torch.mm(r_att_h, self.Lue_embeddings.weight)/2+r_head/2

        r_sim_t = torch.mm(r_tail, self.Lue_embeddings.weight.t())
        r_att_t = self.softmax(r_sim_t)
        r_att_tail = torch.mm(r_att_t, self.Lue_embeddings.weight)/2+r_tail/2

        r_sim_r = torch.mm(r_rel, self.Lur_embeddings.weight.t())
        r_att_r = self.softmax(r_sim_r)
        r_att_rel = torch.mm(r_att_r, self.Lur_embeddings.weight)/2+r_rel/2

        r_triple = r_att_head + r_att_rel - r_att_tail
        regul = F.cosine_similarity(triple, r_triple, dim=-1)

        all_r_h = data['r_h']
        all_r_t = data['r_t']
        all_r_r = data['r_r']
        all_r_head = self.ent_embeddings(all_r_h)
        all_r_tail = self.ent_embeddings(all_r_t)
        all_r_rel = self.rel_embeddings(all_r_r)
        all_r_sim_h = torch.mm(all_r_head, self.Lue_embeddings.weight.t())
        all_r_att_h = self.softmax(all_r_sim_h)
        all_r_att_head = torch.mm(all_r_att_h, self.Lue_embeddings.weight)/2+all_r_head/2

        all_r_sim_t = torch.mm(all_r_tail, self.Lue_embeddings.weight.t())
        all_r_att_t = self.softmax(all_r_sim_t)
        all_r_att_tail = torch.mm(all_r_att_t, self.Lue_embeddings.weight)/2+all_r_tail/2

        all_r_sim_r = torch.mm(all_r_rel, self.Lur_embeddings.weight.t())
        all_r_att_r = self.softmax(all_r_sim_r)
        all_r_att_rel = torch.mm(all_r_att_r, self.Lur_embeddings.weight)/2+all_r_rel/2

        all_r_triple = all_r_att_head + all_r_att_rel - all_r_att_tail
        triple = triple.unsqueeze(dim=-2)
        # all_r_sim = F.cosine_similarity(triple, all_r_triple, dim=-1).sum(dim=1)
        c_list = []
        j=0
        for i in range(math.ceil(all_r_triple.shape[0] / 1000)):
            a = all_r_triple[1000 * j:1000 * (j + 1), :]
            j += 1
            # print(a)
            c = F.cosine_similarity(triple, a, dim=-1)
            c_list.append(c)

        # similarity = F.cosine_similarity(triple,r_triple,dim=-1)
        c_list = torch.cat(c_list,axis=1)
        all_r_sim = torch.exp(c_list).sum(dim=1)
        regul = torch.exp(regul)
        reg = regul/all_r_sim

        score = self._calc(att_head, att_tail, att_rel, mode)
        # score = self._calc(att_head, att_tail, att_rel, mode)
        if self.margin_flag:
            return reg,self.margin - score
        else:
            return reg,score
        # return reg


    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()


    def Get_att(self,ent_id,ent_num):
        if ent_id == None:
            ent_id = [i for i in range(ent_num)]
        ent_id = torch.LongTensor(ent_id)
        ent=self.ent_embeddings(ent_id)

        sim = torch.mm(ent, self.Luent_embeddings.weight.t())
        att = self.softmax(sim)

        # Lu = self.dropout(self.w.weight).unsqueeze(-1).squeeze(0)
        # print(self.dropout(self.w.weight))
        # print(self.w.weight)
        # sim = torch.mm(ent, (Lu*self.latent_unit.weight).t())
        # att = self.softmax(sim)

        print(att[0])
        print(att[588])
        print(att.sum(dim=1))
        # print(att.shape)
        att = att.cpu().detach().numpy()
        # print(att[0][0])
        # print(att[0][83],att[0][84],att[0][85])
        fig = plt.figure(figsize=(200,200))
        sns_plot = sns.heatmap(att)
        plt.show()

    def sim_ana(self,n_h,n_t,n_r,r_h,r_t,r_r,value):

        r_h = torch.LongTensor(r_h)
        r_t = torch.LongTensor(r_t)
        r_r = torch.LongTensor(r_r)

        r_head = self.ent_embeddings(r_h)
        r_tail = self.ent_embeddings(r_t)
        r_rel = self.rel_embeddings(r_r)

        r_sim_h = torch.mm(r_head, self.Lu_embeddings.weight.t())
        r_att_h = self.softmax(r_sim_h)
        r_att_head = torch.mm(r_att_h, self.Lu_embeddings.weight)

        r_sim_t = torch.mm(r_tail, self.Lu_embeddings.weight.t())
        r_att_t = self.softmax(r_sim_t)
        r_att_tail = torch.mm(r_att_t, self.Lu_embeddings.weight)

        r_sim_r = torch.mm(r_rel, self.Lu_embeddings.weight.t())
        r_att_r = self.softmax(r_sim_r)
        r_att_rel = torch.mm(r_att_r, self.Lu_embeddings.weight)

        r_triple = r_att_head + r_att_rel - r_att_tail
        n_h = torch.LongTensor(n_h)
        n_t = torch.LongTensor(n_t)
        n_r = torch.LongTensor(n_r)

        mode = 'normal'
        h = self.ent_embeddings(n_h)
        t = self.ent_embeddings(n_t)
        r = self.rel_embeddings(n_r)
        sim_h = torch.mm(h, self.Lu_embeddings.weight.t())
        att_h = self.softmax(sim_h)
        att_head = torch.mm(att_h, self.Lu_embeddings.weight)

        sim_t = torch.mm(t, self.Lu_embeddings.weight.t())
        att_t = self.softmax(sim_t)
        att_tail = torch.mm(att_t, self.Lu_embeddings.weight)

        sim_r = torch.mm(r, self.Lu_embeddings.weight.t())
        att_r = self.softmax(sim_r)
        att_rel = torch.mm(att_r, self.Lu_embeddings.weight)

        triple = att_head + att_rel - att_tail
        triple = triple.unsqueeze(dim=-2)
        sim = F.cosine_similarity(triple, r_triple, dim=-1)
        # print(m)
        sim=sim.detach().numpy()
        # Topk = torch.topk(sim, 10000)
        # print(Topk)
        num =np.sum(sim>=0.99,axis=1)
        print(num)
        x = [i for i in range(h.shape[0])]
        # plt.plot(x, num)
        plt.bar(x, num, color='b')
        plt.xticks(rotation=90)  # 横坐标每个值旋转90度
        # plt.xlabel('Month')
        # plt.ylabel('Unemployment Rate')
        # plt.title('Monthly Unemployment Trends, 1948')
        plt.show()


    def sim_ana3(self,n_h,n_t,n_r,m_h,m_t,m_r):

        m_h = torch.LongTensor(m_h)
        m_t = torch.LongTensor(m_t)
        m_r = torch.LongTensor(m_r)

        r_head = self.ent_embeddings(m_h)
        r_tail = self.ent_embeddings(m_t)
        r_rel = self.rel_embeddings(m_r)

        r_sim_h = torch.mm(r_head, self.Lu_embeddings.weight.t())
        r_att_h = self.softmax(r_sim_h)
        r_att_head = torch.mm(r_att_h, self.Lu_embeddings.weight)/2+r_head/2

        r_sim_t = torch.mm(r_tail, self.Lu_embeddings.weight.t())
        r_att_t = self.softmax(r_sim_t)
        r_att_tail = torch.mm(r_att_t, self.Lu_embeddings.weight)/2+r_tail/2

        r_sim_r = torch.mm(r_rel, self.Lu_embeddings.weight.t())
        r_att_r = self.softmax(r_sim_r)
        r_att_rel = torch.mm(r_att_r, self.Lu_embeddings.weight)/2+r_rel/2

        r_triple = r_att_head + r_att_rel - r_att_tail
        # r_triple1 = r_triple[:400000, :]
        # r_triple2 = r_triple[400000:, :]
        # r_triple1 = r_triple1.view(4000, -1, self.dim)
        # r_triple1 = r_triple1.unsqueeze(dim=1)
        n_h = torch.LongTensor(n_h)
        n_t = torch.LongTensor(n_t)
        n_r = torch.LongTensor(n_r)

        mode = 'normal'
        h = self.ent_embeddings(n_h)
        t = self.ent_embeddings(n_t)
        r = self.rel_embeddings(n_r)
        sim_h = torch.mm(h, self.Lu_embeddings.weight.t())
        att_h = self.softmax(sim_h)
        att_head = torch.mm(att_h, self.Lu_embeddings.weight)/2+h/2

        sim_t = torch.mm(t, self.Lu_embeddings.weight.t())
        att_t = self.softmax(sim_t)
        att_tail = torch.mm(att_t, self.Lu_embeddings.weight)/2+t/2

        sim_r = torch.mm(r, self.Lu_embeddings.weight.t())
        att_r = self.softmax(sim_r)
        att_rel = torch.mm(att_r, self.Lu_embeddings.weight)/2+r/2

        triple = att_head + att_rel - att_tail
        # triple = triple.unsqueeze(dim=-2)
        sim = F.cosine_similarity(triple, r_triple, dim=-1)
        # print(m)
        sim=sim.detach().numpy()
        # Topk = torch.topk(sim, 10000)
        # print(Topk)
        sim = pd.DataFrame(sim)
        sim.columns=["EL_sim"]
        sim.to_csv("./result/FB15K/real/EL_sim.txt",index=None)
        print(sim)
    def sim_ana4(self,n_h,n_t,n_r,r_h,r_t,r_r,m_h,m_t,m_r):

        r_h = torch.LongTensor(r_h)
        r_t = torch.LongTensor(r_t)
        r_r = torch.LongTensor(r_r)
        r_head = self.ent_embeddings(r_h)
        r_tail = self.ent_embeddings(r_t)
        r_rel = self.rel_embeddings(r_r)
        r_sim_h = torch.mm(r_head, self.Lu_embeddings.weight.t())
        r_att_h = self.softmax(r_sim_h)
        r_att_head = torch.mm(r_att_h, self.Lu_embeddings.weight)/2+r_head/2

        r_sim_t = torch.mm(r_tail, self.Lu_embeddings.weight.t())
        r_att_t = self.softmax(r_sim_t)
        r_att_tail = torch.mm(r_att_t, self.Lu_embeddings.weight)/2+r_tail/2

        r_sim_r = torch.mm(r_rel, self.Lu_embeddings.weight.t())
        r_att_r = self.softmax(r_sim_r)
        r_att_rel = torch.mm(r_att_r, self.Lu_embeddings.weight)/2+r_rel/2
        r_triple = r_att_head + r_att_rel - r_att_tail             #所有的冗余的三元组

        n_h = torch.LongTensor(n_h)
        n_t = torch.LongTensor(n_t)
        n_r = torch.LongTensor(n_r)
        h = self.ent_embeddings(n_h)
        t = self.ent_embeddings(n_t)
        r = self.rel_embeddings(n_r)
        n_sim_h = torch.mm(h, self.Lu_embeddings.weight.t())
        n_att_h = self.softmax(n_sim_h)
        n_att_head = torch.mm(n_att_h, self.Lu_embeddings.weight)/2+h/2

        n_sim_t = torch.mm(t, self.Lu_embeddings.weight.t())
        n_att_t = self.softmax(n_sim_t)
        n_att_tail = torch.mm(n_att_t, self.Lu_embeddings.weight)/2+t/2

        n_sim_r = torch.mm(r, self.Lu_embeddings.weight.t())
        n_att_r = self.softmax(n_sim_r)
        n_att_rel = torch.mm(n_att_r, self.Lu_embeddings.weight)/2+r/2
        triple = n_att_head + n_att_rel - n_att_tail            #所有的非冗余的三元组

        map_h = torch.LongTensor(m_h)
        map_t = torch.LongTensor(m_t)
        map_r = torch.LongTensor(m_r)

        map_head = self.ent_embeddings(map_h)
        map_tail = self.ent_embeddings(map_t)
        map_rel = self.rel_embeddings(map_r)
        m_sim_h = torch.mm(map_head, self.Lu_embeddings.weight.t())
        m_att_h = self.softmax(m_sim_h)
        m_att_head = torch.mm(m_att_h, self.Lu_embeddings.weight)/2+map_head/2

        m_sim_t = torch.mm(map_tail, self.Lu_embeddings.weight.t())
        m_att_t = self.softmax(m_sim_t)
        m_att_tail = torch.mm(m_att_t, self.Lu_embeddings.weight)/2+map_tail/2

        m_sim_r = torch.mm(map_rel, self.Lu_embeddings.weight.t())
        m_att_r = self.softmax(m_sim_r)
        m_att_rel = torch.mm(m_att_r, self.Lu_embeddings.weight)/2+map_rel/2
        map_triple = m_att_head + m_att_rel - m_att_tail
        map_sim = F.cosine_similarity(triple, map_triple, dim=-1)
        EL_sim=map_sim.detach().numpy()

        EL_sim = pd.DataFrame(EL_sim)
        EL_sim.columns=["EL_sim"]

        triple = triple.unsqueeze(dim=-2)
        all_sim = F.cosine_similarity(triple, r_triple, dim=-1)
        # print(m)
        all_sim=all_sim.detach().numpy()

        print(all_sim)
        print(all_sim.shape)
        print(map_sim)
        print(map_sim.shape)
        ranks = []
        for i in range(1000):
            print(i)
            m_sim = map_sim[i]
            a_sim = all_sim[i]
            a_sim = a_sim.tolist()
            # print(a_sim)
            # print(len(a_sim))
            # print(a_sim.sort(reverse=True))
            # sort_a_sim = a_sim.sort(reverse=True)
            a_sim.sort(reverse=True)
            rank = a_sim.index(m_sim.item())
            ranks.append(rank)
        ranks = np.array(ranks)
        ranks = pd.DataFrame(ranks)
        ranks.columns=["EL_sim_rank"]
        EL_sim = pd.concat([EL_sim,ranks],axis=1)
        EL_sim.to_csv("./result/FB15K/real/EL_sim_all.txt",index=None)

    def Get_sim(self,n_h,n_t,n_r,r_h,r_t,r_r):
        r_h = torch.LongTensor(r_h)
        r_t = torch.LongTensor(r_t)
        r_r = torch.LongTensor(r_r)

        r_head = self.ent_embeddings(r_h)
        r_tail = self.ent_embeddings(r_t)
        r_rel = self.rel_embeddings(r_r)

        r_sim_h = torch.mm(r_head, self.Lu_embeddings.weight.t())
        r_att_h = self.softmax(r_sim_h)
        r_att_head = torch.mm(r_att_h, self.Lu_embeddings.weight)/2+r_head/2

        r_sim_t = torch.mm(r_tail, self.Lu_embeddings.weight.t())
        r_att_t = self.softmax(r_sim_t)
        r_att_tail = torch.mm(r_att_t, self.Lu_embeddings.weight)/2+r_tail/2

        r_sim_r = torch.mm(r_rel, self.Lu_embeddings.weight.t())
        r_att_r = self.softmax(r_sim_r)
        r_att_rel = torch.mm(r_att_r, self.Lu_embeddings.weight)/2+r_rel/2

        r_triple = r_att_head + r_att_rel - r_att_tail
        n_h = torch.LongTensor(n_h)
        n_t = torch.LongTensor(n_t)
        n_r = torch.LongTensor(n_r)

        mode = 'normal'
        h = self.ent_embeddings(n_h)
        t = self.ent_embeddings(n_t)
        r = self.rel_embeddings(n_r)
        sim_h = torch.mm(h, self.Lu_embeddings.weight.t())
        att_h = self.softmax(sim_h)
        att_head = torch.mm(att_h, self.Lu_embeddings.weight)/2+h/2

        sim_t = torch.mm(t, self.Lu_embeddings.weight.t())
        att_t = self.softmax(sim_t)
        att_tail = torch.mm(att_t, self.Lu_embeddings.weight)/2+t/2

        sim_r = torch.mm(r, self.Lu_embeddings.weight.t())
        att_r = self.softmax(sim_r)
        att_rel = torch.mm(att_r, self.Lu_embeddings.weight)/2+r/2

        triple = att_head + att_rel - att_tail
        triple = triple.unsqueeze(dim=-2)
        sim = F.cosine_similarity(triple, r_triple, dim=-1)
        # print(m)
        sim=sim.detach().numpy()
        max_index = np.argmax(sim,axis=1)
        print(max_index)
        print(max_index.shape)
        sim_triple = pd.DataFrame(columns=['h','t','r','sim_h','sim_t','sim_r'])
        # for i in [97809,96596]:
        for i in range(len(max_index)):
            print(i)
            new = pd.DataFrame({'h':n_h[i].item(),
                                't':n_t[i].item(),
                                'r':n_r[i].item(),
                                'sim_h': r_h[max_index[i]].item(),
                                'sim_t': r_t[max_index[i]].item(),
                                'sim_r': r_r[max_index[i]].item(),
                                },index = [0])
            # print(new)
            sim_triple = sim_triple.append(new,ignore_index=True)
        print(sim_triple)
        sim_triple.to_csv("./result/FB15K/real/EL_sim_triple.txt", index=None)


