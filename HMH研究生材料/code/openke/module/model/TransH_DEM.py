import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import math
import pandas as pd
from torch.autograd import Variable


class TransH_DEM(Model):

	def __init__(self, ent_tot, rel_tot, dim = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None,
				 Lue_num=300,Lur_num=300,none_map=None):
		super(TransH_DEM, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.none_map = none_map
		self.Lue_num = Lue_num
		self.Lur_num = Lur_num
		self.softmax = nn.Softmax(dim=1)
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
		self.norm_vector = nn.Embedding(self.rel_tot, self.dim)
		self.Lue_embeddings = nn.Embedding(self.Lue_num, self.dim)
		self.Lur_embeddings = nn.Embedding(self.Lur_num, self.dim)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			nn.init.xavier_uniform_(self.norm_vector.weight.data)
			nn.init.xavier_uniform_(self.Lue_embeddings.weight.data)
			nn.init.xavier_uniform_(self.Lur_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.norm_vector.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
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

	def _transfer(self, e, norm):
		norm = F.normalize(norm, p = 2, dim = -1)
		if e.shape[0] != norm.shape[0]:
			e = e.view(-1, norm.shape[0], e.shape[-1])
			norm = norm.view(-1, norm.shape[0], norm.shape[-1])
			e = e - torch.sum(e * norm, -1, True) * norm
			return e.view(-1, e.shape[-1])
		else:
			return e - torch.sum(e * norm, -1, True) * norm

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_norm = self.norm_vector(batch_r)
		h = self._transfer(h, r_norm)
		t = self._transfer(t, r_norm)

		sim_h = torch.mm(h, self.Lue_embeddings.weight.t())
		att_h = self.softmax(sim_h)
		att_head = torch.mm(att_h, self.Lue_embeddings.weight) / 2 + h / 2

		sim_t = torch.mm(t, self.Lue_embeddings.weight.t())
		att_t = self.softmax(sim_t)
		att_tail = torch.mm(att_t, self.Lue_embeddings.weight) / 2 + t / 2

		sim_r = torch.mm(r, self.Lur_embeddings.weight.t())
		att_r = self.softmax(sim_r)
		att_rel = torch.mm(att_r, self.Lur_embeddings.weight) / 2 + r / 2
		score = self._calc(att_head ,att_tail, att_rel, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score


	def Regularization(self, data, batch_size):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		raw_triple = torch.cat(
			[batch_h[:batch_size].unsqueeze(1), batch_t[:batch_size].unsqueeze(1), batch_r[:batch_size].unsqueeze(1)],
			dim=1)
		raw_triple = raw_triple.cpu().numpy()
		raw_triple = pd.DataFrame(raw_triple, columns=["h", "t", "r"])
		# print(raw_triple)
		map_triple = pd.merge(raw_triple, self.none_map)
		map_triple = map_triple[['map_h', 'map_t', 'map_r']]
		# print(map_triple)

		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_norm = self.norm_vector(batch_r)
		h = self._transfer(h, r_norm)
		t = self._transfer(t, r_norm)

		sim_h = torch.mm(h, self.Lue_embeddings.weight.t())
		att_h = self.softmax(sim_h)
		att_head = torch.mm(att_h, self.Lue_embeddings.weight) / 2 + h / 2

		sim_t = torch.mm(t, self.Lue_embeddings.weight.t())
		att_t = self.softmax(sim_t)
		att_tail = torch.mm(att_t, self.Lue_embeddings.weight) / 2 + t / 2

		sim_r = torch.mm(r, self.Lur_embeddings.weight.t())
		att_r = self.softmax(sim_r)
		att_rel = torch.mm(att_r, self.Lur_embeddings.weight) / 2 + r / 2

		triple = att_head[:batch_size, :] + att_rel[:batch_size, :] - att_tail[:batch_size, :]
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
		r_r_norm = self.norm_vector(r_r)
		r_head = self._transfer(r_head, r_r_norm)
		r_tail = self._transfer(r_tail, r_r_norm)

		r_sim_h = torch.mm(r_head, self.Lue_embeddings.weight.t())
		r_att_h = self.softmax(r_sim_h)
		r_att_head = torch.mm(r_att_h, self.Lue_embeddings.weight) / 2 + r_head / 2

		r_sim_t = torch.mm(r_tail, self.Lue_embeddings.weight.t())
		r_att_t = self.softmax(r_sim_t)
		r_att_tail = torch.mm(r_att_t, self.Lue_embeddings.weight) / 2 + r_tail / 2

		r_sim_r = torch.mm(r_rel, self.Lur_embeddings.weight.t())
		r_att_r = self.softmax(r_sim_r)
		r_att_rel = torch.mm(r_att_r, self.Lur_embeddings.weight) / 2 + r_rel / 2

		r_triple = r_att_head + r_att_rel - r_att_tail
		regul = F.cosine_similarity(triple, r_triple, dim=-1)

		all_r_h = data['r_h']
		all_r_t = data['r_t']
		all_r_r = data['r_r']
		all_r_head = self.ent_embeddings(all_r_h)
		all_r_tail = self.ent_embeddings(all_r_t)
		all_r_rel = self.rel_embeddings(all_r_r)
		all_r_norm = self.norm_vector(all_r_r)
		all_r_head = self._transfer(all_r_head, all_r_norm)
		all_r_tail = self._transfer(all_r_tail, all_r_norm)

		all_r_sim_h = torch.mm(all_r_head, self.Lue_embeddings.weight.t())
		all_r_att_h = self.softmax(all_r_sim_h)
		all_r_att_head = torch.mm(all_r_att_h, self.Lue_embeddings.weight) / 2 + all_r_head / 2

		all_r_sim_t = torch.mm(all_r_tail, self.Lue_embeddings.weight.t())
		all_r_att_t = self.softmax(all_r_sim_t)
		all_r_att_tail = torch.mm(all_r_att_t, self.Lue_embeddings.weight) / 2 + all_r_tail / 2

		all_r_sim_r = torch.mm(all_r_rel, self.Lur_embeddings.weight.t())
		all_r_att_r = self.softmax(all_r_sim_r)
		all_r_att_rel = torch.mm(all_r_att_r, self.Lur_embeddings.weight) / 2 + all_r_rel / 2

		all_r_triple = all_r_att_head + all_r_att_rel - all_r_att_tail
		triple = triple.unsqueeze(dim=-2)
		# all_r_sim = F.cosine_similarity(triple, all_r_triple, dim=-1).sum(dim=1)
		c_list = []
		j = 0
		for i in range(math.ceil(all_r_triple.shape[0] / 1000)):
			a = all_r_triple[1000 * j:1000 * (j + 1), :]
			j += 1
			# print(a)
			c = F.cosine_similarity(triple, a, dim=-1)
			c_list.append(c)

		# similarity = F.cosine_similarity(triple,r_triple,dim=-1)
		c_list = torch.cat(c_list, axis=1)
		all_r_sim = torch.exp(c_list).sum(dim=1)
		regul = torch.exp(regul)
		reg = regul / all_r_sim

		score = self._calc(att_head, att_tail, att_rel, mode)
		# score = self._calc(att_head, att_tail, att_rel, mode)
		if self.margin_flag:
			return reg, self.margin - score
		else:
			return reg, score


	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		r_norm = self.norm_vector(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) +
				 torch.mean(r_norm ** 2)) / 4
		return regul
	
	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()