from .Strategy import Strategy

class NegativeSampling(Strategy):

	def __init__(self, model = None, loss = None,regul_rate = 0.0, l3_regul_rate = 0.0, neg_ent=5):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		# self.batch_size = batch_size
		# self.rebatch_size = rebatch_size
		# self.nobatch_size = nobatch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate
		self.neg_ent = neg_ent

	def _get_positive_score(self, score):
		# if step=="E":
		# 	self.batch_size = self.rebatch_size
		# else:
		# 	self.batch_size = self.nobatch_size
		# print("score.shape[0]",score.shape[0])
		# print("self.neg_ent+1", self.neg_ent+1)
		batch_size = int(score.shape[0]/(self.neg_ent+1))
		positive_score = score[:batch_size]
		positive_score = positive_score.view(-1, batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		# if step=="E":
		# 	self.batch_size = self.rebatch_size
		# else:
		# 	self.batch_size = self.nobatch_size
		batch_size = int(score.shape[0]/(self.neg_ent+1))
		negative_score = score[batch_size:]
		negative_score = negative_score.view(-1, batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		score = self.model(data)
		# print(score.shape[0])
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)
		loss_res = self.loss(p_score, n_score)
		# if data['regul'] == True:
		# 	loss_res += self.regul_rate * self.model.Regularization(data)
		# if self.regul_rate != 0:
		# 	loss_res += self.regul_rate * self.model.regularization(data)
		# if self.l3_regul_rate != 0:
		# 	loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res