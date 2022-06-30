import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import math
import pandas as pd
from torch.autograd import Variable

class TransD_DEM(Model):

	def __init__(self, ent_tot, rel_tot, dim_e = 100, dim_r = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None,
				 Lue_num=300,Lur_num=300,none_map=None):
		super(TransD_DEM, self).__init__(ent_tot, rel_tot)
		
		self.dim_e = dim_e
		self.dim_r = dim_r
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm
		self.none_map = none_map
		self.Lue_num = Lue_num
		self.Lur_num = Lur_num
		self.softmax = nn.Softmax(dim=1)
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
		self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)
		self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)
		self.Lue_embeddings = nn.Embedding(self.Lue_num, self.dim_e)
		self.Lur_embeddings = nn.Embedding(self.Lur_num, self.dim_r)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			nn.init.xavier_uniform_(self.ent_transfer.weight.data)
			nn.init.xavier_uniform_(self.rel_transfer.weight.data)
			nn.init.xavier_uniform_(self.Lue_embeddings.weight.data)
			nn.init.xavier_uniform_(self.Lur_embeddings.weight.data)
		else:
			self.ent_embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False
			)
			self.rel_embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.ent_embedding_range.item(), 
				b = self.ent_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.rel_embedding_range.item(), 
				b= self.rel_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.ent_transfer.weight.data, 
				a= -self.ent_embedding_range.item(), 
				b= self.ent_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_transfer.weight.data, 
				a= -self.rel_embedding_range.item(), 
				b= self.rel_embedding_range.item()
			)
		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	def _resize(self, tensor, axis, size):
		shape = tensor.size()
		osize = shape[axis]
		if osize == size:
			return tensor
		if (osize > size):
			return torch.narrow(tensor, axis, 0, size)
		paddings = []
		for i in range(len(shape)):
			if i == axis:
				paddings = [0, size - osize] + paddings
			else:
				paddings = [0, 0] + paddings
		print (paddings)
		return F.pad(tensor, paddings = paddings, mode = "constant", value = 0)

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

	def _transfer(self, e, e_transfer, r_transfer):
		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], e.shape[-1])
			e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
			r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
			e = F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)			
			return e.view(-1, e.shape[-1])
		else:
			return F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		h = self._transfer(h, h_transfer, r_transfer)
		t = self._transfer(t, t_transfer, r_transfer)

		sim_h = torch.mm(h, self.Lue_embeddings.weight.t())
		att_h = self.softmax(sim_h)
		att_head = torch.mm(att_h, self.Lue_embeddings.weight) / 2 + h / 2

		sim_t = torch.mm(t, self.Lue_embeddings.weight.t())
		att_t = self.softmax(sim_t)
		att_tail = torch.mm(att_t, self.Lue_embeddings.weight) / 2 + t / 2

		sim_r = torch.mm(r, self.Lur_embeddings.weight.t())
		att_r = self.softmax(sim_r)
		att_rel = torch.mm(att_r, self.Lur_embeddings.weight) / 2 + r / 2

		# score = self._calc(h ,t, r, mode)
		score = self._calc(att_head, att_tail, att_rel, mode)
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
		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		h = self._transfer(h, h_transfer, r_transfer)
		t = self._transfer(t, t_transfer, r_transfer)
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
		r_h_transfer = self.ent_transfer(r_h)
		r_t_transfer = self.ent_transfer(r_t)
		r_r_transfer = self.rel_transfer(r_r)
		r_head = self._transfer(r_head, r_h_transfer, r_r_transfer)
		r_tail = self._transfer(r_tail, r_t_transfer, r_r_transfer)

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
		all_r_h_transfer = self.ent_transfer(all_r_h)
		all_r_t_transfer = self.ent_transfer(all_r_t)
		all_r_r_transfer = self.rel_transfer(all_r_r)
		all_r_head = self._transfer(all_r_head, all_r_h_transfer, all_r_r_transfer)
		all_r_tail = self._transfer(all_r_tail, all_r_t_transfer, all_r_r_transfer)

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
		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) + 
				 torch.mean(h_transfer ** 2) + 
				 torch.mean(t_transfer ** 2) + 
				 torch.mean(r_transfer ** 2)) / 6
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()