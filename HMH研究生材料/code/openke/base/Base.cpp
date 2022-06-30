#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <pthread.h>

extern "C"
void setInPath(char *path);

extern "C"
void setRedunPath(char *path);

extern "C"
void setTrainPath(char *path);

extern "C"
void setValidPath(char *path);

extern "C"
void setTestPath(char *path);

extern "C"
void setEntPath(char *path);

extern "C"
void setRelPath(char *path);

extern "C"
void setOutPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
void setBern(INT con);

extern "C"
INT getWorkThreads();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getRedunTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

extern "C"
void randReset();

extern "C"
void importTrainFiles();

//struct Parameter {
//	INT id;
//	INT *batch_h1;
//	INT *batch_t1;
//	INT *batch_r1;
//	REAL *batch_y1;
//	INT batchSize1;
//	INT *batch_h2;
//	INT *batch_t2;
//	INT *batch_r2;
//	REAL *batch_y2;
//	INT batchSize2;
//	INT negRate;
//	INT negRelRate;
//	bool p;
//	bool val_loss;
//	INT mode;
//	bool filter_flag;
//};

struct Parameter {
	INT id;
	INT *batch_h_train;
	INT *batch_t_train;
	INT *batch_r_train;
	REAL *batch_y_train;
	INT batchSize_train;
	INT *batch_h_redun;
	INT *batch_t_redun;
	INT *batch_r_redun;
	REAL *batch_y_redun;
	INT batchSize_redun;
	INT negRate;
	INT negRelRate;
	bool p;
	bool val_loss;
	INT mode;
	bool filter_flag;
};

void* getBatch_redun(void* con) {
	Parameter *para = (Parameter *)(con);
	INT id = para -> id;
	INT *batch_h_redun = para -> batch_h_redun;
	INT *batch_t_redun = para -> batch_t_redun;
	INT *batch_r_redun = para -> batch_r_redun;
	REAL *batch_y_redun = para -> batch_y_redun;
	INT batchSize_redun = para -> batchSize_redun;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	bool p = para -> p;
	bool val_loss = para -> val_loss;
	INT mode = para -> mode;
	bool filter_flag = para -> filter_flag;
	INT lef, rig;
	if (batchSize_redun % workThreads == 0) {
		lef = id * (batchSize_redun / workThreads);
		rig = (id + 1) * (batchSize_redun / workThreads);
	} else {
		lef = id * (batchSize_redun / workThreads + 1);
		rig = (id + 1) * (batchSize_redun / workThreads + 1);
		if (rig > batchSize_redun) rig = batchSize_redun;
	}
	REAL prob = 500;
	if (val_loss == false) {
		for (INT batch_redun = lef; batch_redun < rig; batch_redun++) {
			INT i = rand_max(id, redunTotal);
			batch_h_redun[batch_redun] = redunList[i].h;
			batch_t_redun[batch_redun] = redunList[i].t;
			batch_r_redun[batch_redun] = redunList[i].r;
//			printf("r%ld\n",redunList[i].r);
			batch_y_redun[batch_redun] = 1;
			INT last = batchSize_redun;
			for (INT times = 0; times < negRate; times ++) {
				if (mode == 0){
					if (bernFlag)
						prob = 1000 * redunright_mean[redunList[i].r] / (redunright_mean[redunList[i].r] + redunleft_mean[redunList[i].r]);
					if (randd(id) % 1000 < prob) {
						batch_h_redun[batch_redun + last] = redunList[i].h;
						batch_t_redun[batch_redun + last] = corrupt_head_redun(id, redunList[i].h, redunList[i].r);
						batch_r_redun[batch_redun + last] = redunList[i].r;
					} else {
						batch_h_redun[batch_redun + last] = corrupt_tail_redun(id, redunList[i].t, redunList[i].r);
						batch_t_redun[batch_redun + last] = redunList[i].t;
						batch_r_redun[batch_redun + last] = redunList[i].r;
					}
					batch_y_redun[batch_redun + last] = -1;
					last += batchSize_redun;
				} else {
					if(mode == -1){
						batch_h_redun[batch_redun + last] = corrupt_tail_redun(id, redunList[i].t, redunList[i].r);
						batch_t_redun[batch_redun + last] = redunList[i].t;
						batch_r_redun[batch_redun + last] = redunList[i].r;
					} else {
						batch_h_redun[batch_redun + last] = redunList[i].h;
						batch_t_redun[batch_redun + last] = corrupt_head_redun(id, redunList[i].h, redunList[i].r);
						batch_r_redun[batch_redun + last] = redunList[i].r;
					}
					batch_y_redun[batch_redun + last] = -1;
					last += batchSize_redun;
				}
			}
			for (INT times = 0; times < negRelRate; times++) {
				batch_h_redun[batch_redun + last] = redunList[i].h;
				batch_t_redun[batch_redun + last] = redunList[i].t;
				batch_r_redun[batch_redun + last] = corrupt_rel_redun(id, redunList[i].h, redunList[i].t, redunList[i].r, p);
				batch_y_redun[batch_redun + last] = -1;
				last += batchSize_redun;
			}
		}
	}
	else
	{
		for (INT batch_redun = lef; batch_redun < rig; batch_redun++)
		{
			batch_h_redun[batch_redun] = validList[batch_redun].h;
			batch_t_redun[batch_redun] = validList[batch_redun].t;
			batch_r_redun[batch_redun] = validList[batch_redun].r;
			batch_y_redun[batch_redun] = 1;
		}
	}
	pthread_exit(NULL);
}

void* getBatch_train(void* con) {
	Parameter *para = (Parameter *)(con);
	INT id = para -> id;
	INT *batch_h_train = para -> batch_h_train;
	INT *batch_t_train = para -> batch_t_train;
	INT *batch_r_train = para -> batch_r_train;
	REAL *batch_y_train = para -> batch_y_train;
	INT batchSize_train = para -> batchSize_train;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	bool p = para -> p;
	bool val_loss = para -> val_loss;
	INT mode = para -> mode;
	bool filter_flag = para -> filter_flag;
	INT lef, rig;
	if (batchSize_train % workThreads == 0) {
		lef = id * (batchSize_train / workThreads);
		rig = (id + 1) * (batchSize_train / workThreads);
	} else {
		lef = id * (batchSize_train / workThreads + 1);
		rig = (id + 1) * (batchSize_train / workThreads + 1);
		if (rig > batchSize_train) rig = batchSize_train;
	}
	REAL prob = 500;
	if (val_loss == false) {
		for (INT batch_train = lef; batch_train < rig; batch_train++) {
			INT i = rand_max(id, trainTotal);
			batch_h_train[batch_train] = trainList[i].h;
			batch_t_train[batch_train] = trainList[i].t;
			batch_r_train[batch_train] = trainList[i].r;
			batch_y_train[batch_train] = 1;
			INT last = batchSize_train;
			for (INT times = 0; times < negRate; times ++) {
				if (mode == 0){
					if (bernFlag)
						prob = 1000 * trainright_mean[trainList[i].r] / (trainright_mean[trainList[i].r] + trainleft_mean[trainList[i].r]);
					if (randd(id) % 1000 < prob) {
						batch_h_train[batch_train + last] = trainList[i].h;
						batch_t_train[batch_train + last] = corrupt_head_train(id, trainList[i].h, trainList[i].r);
						batch_r_train[batch_train + last] = trainList[i].r;
					} else {
						batch_h_train[batch_train + last] = corrupt_tail_train(id, trainList[i].t, trainList[i].r);
						batch_t_train[batch_train + last] = trainList[i].t;
						batch_r_train[batch_train + last] = trainList[i].r;
					}
					batch_y_train[batch_train + last] = -1;
					last += batchSize_train;
				} else {
					if(mode == -1){
						batch_h_train[batch_train + last] = corrupt_tail_train(id, trainList[i].t, trainList[i].r);
						batch_t_train[batch_train + last] = trainList[i].t;
						batch_r_train[batch_train + last] = trainList[i].r;
					} else {
						batch_h_train[batch_train + last] = trainList[i].h;
						batch_t_train[batch_train + last] = corrupt_head_train(id, trainList[i].h, trainList[i].r);
						batch_r_train[batch_train + last] = trainList[i].r;
					}
					batch_y_train[batch_train + last] = -1;
					last += batchSize_train;
				}
			}
			for (INT times = 0; times < negRelRate; times++) {
				batch_h_train[batch_train + last] = trainList[i].h;
				batch_t_train[batch_train + last] = trainList[i].t;
				batch_r_train[batch_train + last] = corrupt_rel_train(id, trainList[i].h, trainList[i].t, trainList[i].r, p);
				batch_y_train[batch_train + last] = -1;
				last += batchSize_train;
			}
		}
	}
	else
	{
		for (INT batch_train = lef; batch_train < rig; batch_train++)
		{
			batch_h_train[batch_train] = validList[batch_train].h;
			batch_t_train[batch_train] = validList[batch_train].t;
			batch_r_train[batch_train] = validList[batch_train].r;
			batch_y_train[batch_train] = 1;
		}
	}
	pthread_exit(NULL);
}

extern "C"
void sampling_redun(
		INT *batch_h_redun,
		INT *batch_t_redun,
		INT *batch_r_redun,
		REAL *batch_y_redun,
		INT batchSize_redun,
		INT negRate = 1, 
		INT negRelRate = 0, 
		INT mode = 0,
		bool filter_flag = true,
		bool p = false, 
		bool val_loss = false
) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));

	for (INT threads = 0; threads < workThreads; threads++) {
//	    printf("%ld\n",threads);
		para[threads].id = threads;
		para[threads].batch_h_redun = batch_h_redun;
		para[threads].batch_t_redun = batch_t_redun;
		para[threads].batch_r_redun = batch_r_redun;
		para[threads].batch_y_redun = batch_y_redun;
		para[threads].batchSize_redun = batchSize_redun;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		para[threads].p = p;
		para[threads].val_loss = val_loss;
		para[threads].mode = mode;
		para[threads].filter_flag = filter_flag;
		pthread_create(&pt[threads], NULL, getBatch_redun, (void*)(para+threads));
	}
//	printf("%d",222);
	for (INT threads = 0; threads < workThreads; threads++)
		pthread_join(pt[threads], NULL);
//		printf("%d",333);

	free(pt);
	free(para);
}

extern "C"
void sampling_train(
		INT *batch_h_train,
		INT *batch_t_train,
		INT *batch_r_train,
		REAL *batch_y_train,
		INT batchSize_train,
		INT negRate = 1,
		INT negRelRate = 0,
		INT mode = 0,
		bool filter_flag = true,
		bool p = false,
		bool val_loss = false
) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
	for (INT threads = 0; threads < workThreads; threads++) {
		para[threads].id = threads;
		para[threads].batch_h_train = batch_h_train;
		para[threads].batch_t_train = batch_t_train;
		para[threads].batch_r_train = batch_r_train;
		para[threads].batch_y_train = batch_y_train;
		para[threads].batchSize_train = batchSize_train;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		para[threads].p = p;
		para[threads].val_loss = val_loss;
		para[threads].mode = mode;
		para[threads].filter_flag = filter_flag;
		pthread_create(&pt[threads], NULL, getBatch_train, (void*)(para+threads));
	}
	for (INT threads = 0; threads < workThreads; threads++)
		pthread_join(pt[threads], NULL);

	free(pt);
	free(para);
}

int main() {
	importTrainFiles();
	return 0;
}