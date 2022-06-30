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
import copy
import time
from tqdm import tqdm
# import matplotlib.pyplot as plt

class TrainerEL(object):

	def __init__(self, 
				 model = None,
				 data = None,
				 train_times = 1000,
				 alpha = 0.5,
				 use_gpu = True,
				 opt_method = "sgd",
				 save_steps = None,
				 checkpoint_dir = None):

		self.work_threads = 8
		self.train_times = train_times

		self.opt_method = opt_method
		self.optimizer = None
		self.lr_decay = 0
		self.weight_decay = 0
		self.alpha = alpha

		self.model = model
		self.data = data
		self.use_gpu = use_gpu
		self.save_steps = save_steps
		self.checkpoint_dir = checkpoint_dir
		# self.r_h = r_h,
		# self.r_t = r_t,
		# self.r_r = r_r


	def train_one_step(self, data,regul,r_h,r_t,r_r):
		self.optimizer.zero_grad()

		loss = self.model({
			'batch_h': self.to_var(data['batch_h'], self.use_gpu),
			'batch_t': self.to_var(data['batch_t'], self.use_gpu),
			'batch_r': self.to_var(data['batch_r'], self.use_gpu),
			'batch_y': self.to_var(data['batch_y'], self.use_gpu),
			'mode': data['mode'],
			'regul':regul,
			'r_h' : self.to_var(r_h, self.use_gpu),
			'r_t' : self.to_var(r_t, self.use_gpu),
			'r_r' : self.to_var(r_r, self.use_gpu)
		}
		)
		loss.backward()
		self.optimizer.step()
		# print(loss)
		# print(loss.item())
		return loss.item()


	def E_step(self, data,regul,r_h,r_t,r_r):
		for name, param in self.model.named_parameters():
			# print("########",name)
			if name == "model.Lue_embeddings.weight" or name == "model.Lur_embeddings.weight":
				param.requires_grad = True
			else:
				param.requires_grad = False
		return self.train_one_step(data,regul,r_h,r_t,r_r)

	def M_step(self, data,regul,r_h,r_t,r_r):
		for name, param in self.model.named_parameters():
			# print(name)
			if name == "model.ent_embeddings.weight" or name == "model.rel_embeddings.weight":
				param.requires_grad = True
			else:
				param.requires_grad = False
		return self.train_one_step(data,regul,r_h,r_t,r_r)

	# def Get_r_emb(self,r_h,r_t,r_r):
	# 	r_h = self.to_var(r_h, self.use_gpu)
	# 	r_t = self.to_var(r_t, self.use_gpu)
	# 	r_r = self.to_var(r_r, self.use_gpu)
	# 	r_head = self.ent_embeddings(r_h)
	# 	r_tail = self.ent_embeddings(r_t)
	# 	r_rel = self.rel_embeddings(r_r)

	def run(self,r_h,r_t,r_r):
		if self.use_gpu:
			self.model.cuda()

		if self.optimizer != None:
			pass
		elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
			self.optimizer = optim.Adagrad(
				self.model.parameters(),
				lr=self.alpha,
				lr_decay=self.lr_decay,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
			self.optimizer = optim.Adadelta(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		elif self.opt_method == "Adam" or self.opt_method == "adam":
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=self.alpha,
				weight_decay=self.weight_decay,
			)
		else:
			self.optimizer = optim.SGD(
				self.model.parameters(),
				lr = self.alpha,
				weight_decay=self.weight_decay,
			)
		# print("Finish initializing...")
		
		training_range = tqdm(range(self.train_times))
		losses1 = []
		losses2 = []

		for epoch in training_range:


			start = time.time()
			# print("time",start)
			redun_res = 0.0
			self.data.judge_way(2)
			regul = False
			for data in self.data:
				loss = self.E_step(data,regul,r_h,r_t,r_r)
				redun_res += loss


			none_res =0.0
			self.data.judge_way(1)
			i=0
			regul = True
			for data in self.data:
				# print(i)
				i+=1
				loss = self.M_step(data,regul,r_h,r_t,r_r)
				none_res += loss

			self.data.judge_way(2)
			regul = False
			for data in self.data:
				loss = self.M_step(data,regul,r_h,r_t,r_r)
				none_res += loss
			end = time.time()
			t = end - start
			print("epoch:",epoch,"time:",t)
			training_range.set_description("Epoch %d | redunloss: %f | allloss: %f" % (epoch, redun_res, none_res))
			# training_range.set_description("Epoch %d | redunloss: %f " % (epoch, redun_res))


			losses1 += [redun_res]
			losses2 += [none_res]
			# print("losses1",losses1)
			# print("losses2", losses2)

			if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
				print("Epoch %d has finished, saving..." % (epoch))
				self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

		# plt.plot(losses,label="loss")
		# plt.legend(loc="best")
		# plt.xlabel("steps")
		# plt.ylabel("Loss")
		# plt.ylim = ((0,0.2))
		# plt.show()

	def set_model(self, model):
		self.model = model

	def to_var(self, x, use_gpu):
		if use_gpu:
			return Variable(torch.from_numpy(x).cuda())
		else:
			return Variable(torch.from_numpy(x))

	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_lr_decay(self, lr_decay):
		self.lr_decay = lr_decay

	def set_weight_decay(self, weight_decay):
		self.weight_decay = weight_decay

	def set_opt_method(self, opt_method):
		self.opt_method = opt_method

	def set_train_times(self, train_times):
		self.train_times = train_times

	def set_save_steps(self, save_steps, checkpoint_dir = None):
		self.save_steps = save_steps
		if not self.checkpoint_dir:
			self.set_checkpoint_dir(checkpoint_dir)

	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir