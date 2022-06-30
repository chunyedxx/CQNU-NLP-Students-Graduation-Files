import openke
from openke.config import TrainerHL, Tester
from openke.module.model import TransH_DEM
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling_DEM
from openke.data import TrainDataLoader, TestDataLoader,RedunData

import torch
import numpy as np
import random
import pandas as pd
none_map = pd.read_csv("./benchmarks/FB15K/none_map.txt",sep=" ")
print(none_map)
r_h,r_t,r_r = RedunData("./benchmarks/FB15K/redun.txt")
# r_h,r_t,r_r = np.array(r_h[:1000]),np.array(r_t[:1000]),np.array(r_r[:1000])
r_h,r_t,r_r = np.array(r_h),np.array(r_t),np.array(r_r)
#


data = TrainDataLoader(
	in_path = "./benchmarks/FB15K/",
	nbatches1 = 300,
	nbatches2 = 300,
	threads = 8,
	sampling_mode = "normal",
	bern_flag = 1,
	filter_flag = 1,
	neg_ent = 5,
	neg_rel = 0,
	data1="none.txt",
	data2="train2id.txt")     #When you test, you need to replace the data2 with "train2id".


test_dataloader = TestDataLoader("./benchmarks/FB15K/", "link")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)

transh_dem = TransH_DEM(
	ent_tot = data.get_ent_tot(),
	rel_tot = data.get_rel_tot(),
	dim = 50,
	p_norm = 1,
	norm_flag = True,
	Lue_num=300,
	Lur_num=200,
	none_map=none_map,
)


model = NegativeSampling_DEM(
	model = transh_dem,
	loss = MarginLoss(margin = 3.0),
	neg_ent=5,
	regul_rate=1
	# batch_size = data.get_batch_size()
)
#
for name, param in transh_dem.named_parameters():
	print(name)



trainer = TrainerHL(model = model, data = data, train_times = 50,opt_method = "sgd", alpha = 0.3, use_gpu = True)
# trainer.run(r_h=r_h,r_t=r_t,r_r=r_r)
# transh_dem.save_checkpoint('./checkpoint/H_DEM11.ckpt')


transh_dem.load_checkpoint('./checkpoint/H_DEM11.ckpt')
tester = Tester(model = transh_dem, data_loader = test_dataloader, use_gpu = True,
				path2="result/FB15K/new/H_DEM11.csv"
				)
tester.run_link_prediction(type_constrain = False)
