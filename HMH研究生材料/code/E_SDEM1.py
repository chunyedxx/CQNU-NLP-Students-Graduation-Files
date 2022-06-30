import openke
from openke.config import TrainerEL, Tester
from openke.module.model import TransE_DEM
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
	data2="redun.txt")   #When you test, you need to replace the data2 with "train2id".


test_dataloader = TestDataLoader("./benchmarks/FB15K/", "link")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)

transe_dem = TransE_DEM(
	ent_tot = data.get_ent_tot(),
	rel_tot = data.get_rel_tot(),
	dim =50,
	p_norm = 1,
	norm_flag = True,
	Lue_num=300,
	Lur_num=200,
	none_map=none_map,
)


model = NegativeSampling_DEM(
	model = transe_dem,
	loss = MarginLoss(margin = 3.0),
	neg_ent=5,
	regul_rate=1
	# batch_size = data.get_batch_size()
)
#
for name, param in model.named_parameters():
	print(name)



trainer = TrainerEL(model = model, data = data, train_times = 50,opt_method = "sgd", alpha = 0.3, use_gpu = True)
trainer.run(r_h=r_h,r_t=r_t,r_r=r_r)
transe_dem.save_checkpoint('./checkpoint1/E_DEM12.ckpt')


# transe_dem.load_checkpoint('./checkpoint1/E_DEM12.ckpt')
# tester = Tester(model = transe_dem, data_loader = test_dataloader, use_gpu = True,
# 				path2="./result/FB15K/new/E_DEM12.csv"
# 				)
# tester.run_link_prediction(type_constrain = False)


# mr ['224.54049'] fmr ['129.90369'] hit10 ['0.48994'] fhit10 ['0.64265'] a 100dim 5neg
# mr ['246.92556'] fmr ['153.06846'] hit10 ['0.41051'] fhit10 ['0.52757'] 1 300 200 adam 50dim
# mr ['245.83574'] fmr ['152.91435'] hit10 ['0.40803'] fhit10 ['0.52574'] 2 200 200 adam 50dim
# mr ['242.18512'] fmr ['149.27089'] hit10 ['0.41182'] fhit10 ['0.52895'] 3 200 100 adam 50dim
# mr ['242.52345'] fmr ['148.88008'] hit10 ['0.41113'] fhit10 ['0.53036'] 4 100 100 adam 50dim
# mr ['243.42432'] fmr ['150.09467'] hit10 ['0.40894'] fhit10 ['0.52720'] 5 100 50 adam 50dim
# mr ['214.78496'] fmr ['120.16660'] hit10 ['0.47556'] fhit10 ['0.61870'] 6 300 200 adam 100dim


# mr ['224.80600'] fmr ['129.39892'] hit10 ['0.49790'] fhit10 ['0.65837'] a1 100dim 25neg
# mr ['201.87900'] fmr ['105.02902'] hit10 ['0.50157'] fhit10 ['0.66924'] 7 300 200 adam 100dim 25neg

# 7 300 200 adam 100dim 10neg

