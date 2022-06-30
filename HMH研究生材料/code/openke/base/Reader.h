 #ifndef READER_H
#define READER_H
#include "Setting.h"
#include "Triple.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cmath>

INT *trainfreqRel, *trainfreqEnt;
INT *trainlefHead, *trainrigHead;
INT *trainlefTail, *trainrigTail;
INT *trainlefRel, *trainrigRel;
REAL *trainleft_mean, *trainright_mean;
//REAL *trainprob;
REAL *prob;

INT *redunfreqRel, *redunfreqEnt;
INT *redunlefHead, *redunrigHead;
INT *redunlefTail, *redunrigTail;
INT *redunlefRel, *redunrigRel;
REAL *redunleft_mean, *redunright_mean;
//REAL *redunprob;

Triple *redunList;
Triple *trainList;
Triple *redunHead;
Triple *redunTail;
Triple *redunRel;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;

INT *testLef, *testRig;
INT *validLef, *validRig;

extern "C"
void importProb(REAL temp){
    if (prob != NULL)
        free(prob);
    FILE *fin;
    fin = fopen((inPath + "kl_prob.txt").c_str(), "r");
//    printf("Current temperature:%f\n", temp);
    prob = (REAL *)calloc(relationTotal * (relationTotal - 1), sizeof(REAL));
    INT tmp;
    for (INT i = 0; i < relationTotal * (relationTotal - 1); ++i){
        tmp = fscanf(fin, "%f", &prob[i]);
    }
    REAL sum = 0.0;
    for (INT i = 0; i < relationTotal; ++i) {
        for (INT j = 0; j < relationTotal-1; ++j){
            REAL tmp = exp(-prob[i * (relationTotal - 1) + j] / temp);
            sum += tmp;
            prob[i * (relationTotal - 1) + j] = tmp;
        }
        for (INT j = 0; j < relationTotal-1; ++j){
            prob[i*(relationTotal-1)+j] /= sum;
        }
        sum = 0;
    }
    fclose(fin);
}

extern "C"
void importTrainFiles() {

//	printf("The toolkit is importing datasets.\n");
	FILE *fin;
	int tmp;

    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &relationTotal);
//	printf("The total of relations is %ld.\n", relationTotal);
	fclose(fin);

    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &entityTotal);
//	printf("The total of entities is %ld.\n", entityTotal);
	fclose(fin);

//###################train
    if (train_file == "")
        fin = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        fin = fopen(train_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &trainTotal);
	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainfreqRel = (INT *)calloc(relationTotal, sizeof(INT));
	trainfreqEnt = (INT *)calloc(entityTotal, sizeof(INT));
	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainList[i].h);
		tmp = fscanf(fin, "%ld", &trainList[i].t);
		tmp = fscanf(fin, "%ld", &trainList[i].r);
	}
	fclose(fin);

	std::sort(trainList, trainList + trainTotal, Triple::cmp_head);
	tmp = trainTotal; trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
	trainfreqEnt[trainList[0].t] += 1;
	trainfreqEnt[trainList[0].h] += 1;
	trainfreqRel[trainList[0].r] += 1;
	for (INT i = 1; i < tmp; i++)
		if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r || trainList[i].t != trainList[i - 1].t) {
			trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[trainTotal] = trainList[i];
			trainTotal++;
			trainfreqEnt[trainList[i].t]++;
			trainfreqEnt[trainList[i].h]++;
			trainfreqRel[trainList[i].r]++;
		}

	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_head);
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_tail);
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rel);
//	printf("The total of train triples is %ld.\n", trainTotal);

	trainlefHead = (INT *)calloc(entityTotal, sizeof(INT));
	trainrigHead = (INT *)calloc(entityTotal, sizeof(INT));
	trainlefTail = (INT *)calloc(entityTotal, sizeof(INT));
	trainrigTail = (INT *)calloc(entityTotal, sizeof(INT));
	trainlefRel = (INT *)calloc(entityTotal, sizeof(INT));
	trainrigRel = (INT *)calloc(entityTotal, sizeof(INT));
	memset(trainrigHead, -1, sizeof(INT)*entityTotal);
	memset(trainrigTail, -1, sizeof(INT)*entityTotal);
	memset(trainrigRel, -1, sizeof(INT)*entityTotal);
	for (INT i = 1; i < trainTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			trainrigTail[trainTail[i - 1].t] = i - 1;
			trainlefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			trainrigHead[trainHead[i - 1].h] = i - 1;
			trainlefHead[trainHead[i].h] = i;
		}
		if (trainRel[i].h != trainRel[i - 1].h) {
			trainrigRel[trainRel[i - 1].h] = i - 1;
			trainlefRel[trainRel[i].h] = i;
		}
	}
	trainlefHead[trainHead[0].h] = 0;
	trainrigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	trainlefTail[trainTail[0].t] = 0;
	trainrigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	trainlefRel[trainRel[0].h] = 0;
	trainrigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;

	trainleft_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	trainright_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = trainlefHead[i] + 1; j <= trainrigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				trainleft_mean[trainHead[j].r] += 1.0;
		if (trainlefHead[i] <= trainrigHead[i])
			trainleft_mean[trainHead[trainlefHead[i]].r] += 1.0;
		for (INT j = trainlefTail[i] + 1; j <= trainrigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				trainright_mean[trainTail[j].r] += 1.0;
		if (trainlefTail[i] <= trainrigTail[i])
			trainright_mean[trainTail[trainlefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		trainleft_mean[i] = trainfreqRel[i] / trainleft_mean[i];
		trainright_mean[i] = trainfreqRel[i] / trainright_mean[i];
	}

//###################redun

	if (redun_file == "")
        fin = fopen((inPath + "redun.txt").c_str(), "r");
    else
        fin = fopen(redun_file.c_str(), "r");
    tmp = fscanf(fin, "%ld", &redunTotal);
	redunList = (Triple *)calloc(redunTotal, sizeof(Triple));
	redunHead = (Triple *)calloc(redunTotal, sizeof(Triple));
	redunTail = (Triple *)calloc(redunTotal, sizeof(Triple));
	redunRel = (Triple *)calloc(redunTotal, sizeof(Triple));
	redunfreqRel = (INT *)calloc(relationTotal, sizeof(INT));
	redunfreqEnt = (INT *)calloc(entityTotal, sizeof(INT));
	for (INT i = 0; i < redunTotal; i++) {
		tmp = fscanf(fin, "%ld", &redunList[i].h);
		tmp = fscanf(fin, "%ld", &redunList[i].t);
		tmp = fscanf(fin, "%ld", &redunList[i].r);
	}
	fclose(fin);

	std::sort(redunList, redunList + redunTotal, Triple::cmp_head);
	tmp = redunTotal; redunTotal = 1;
	redunHead[0] = redunTail[0] = redunRel[0] = redunList[0];
	redunfreqEnt[redunList[0].t] += 1;
	redunfreqEnt[redunList[0].h] += 1;
	redunfreqRel[redunList[0].r] += 1;
	for (INT i = 1; i < tmp; i++)
		if (redunList[i].h != redunList[i - 1].h || redunList[i].r != redunList[i - 1].r || redunList[i].t != redunList[i - 1].t) {
			redunHead[redunTotal] = redunTail[redunTotal] = redunRel[redunTotal] = redunList[redunTotal] = redunList[i];
			redunTotal++;
			redunfreqEnt[redunList[i].t]++;
			redunfreqEnt[redunList[i].h]++;
			redunfreqRel[redunList[i].r]++;
		}

	std::sort(redunHead, redunHead + redunTotal, Triple::cmp_head);
	std::sort(redunTail, redunTail + redunTotal, Triple::cmp_tail);
	std::sort(redunRel, redunRel + redunTotal, Triple::cmp_rel);
//	printf("The total of redun triples is %ld.\n", redunTotal);

	redunlefHead = (INT *)calloc(entityTotal, sizeof(INT));
	redunrigHead = (INT *)calloc(entityTotal, sizeof(INT));
	redunlefTail = (INT *)calloc(entityTotal, sizeof(INT));
	redunrigTail = (INT *)calloc(entityTotal, sizeof(INT));
	redunlefRel = (INT *)calloc(entityTotal, sizeof(INT));
	redunrigRel = (INT *)calloc(entityTotal, sizeof(INT));
	memset(redunrigHead, -1, sizeof(INT)*entityTotal);
	memset(redunrigTail, -1, sizeof(INT)*entityTotal);
	memset(redunrigRel, -1, sizeof(INT)*entityTotal);
	for (INT i = 1; i < redunTotal; i++) {
		if (redunTail[i].t != redunTail[i - 1].t) {
			redunrigTail[redunTail[i - 1].t] = i - 1;
			redunlefTail[redunTail[i].t] = i;
		}
		if (redunHead[i].h != redunHead[i - 1].h) {
			redunrigHead[redunHead[i - 1].h] = i - 1;
			redunlefHead[redunHead[i].h] = i;
		}
		if (redunRel[i].h != redunRel[i - 1].h) {
			redunrigRel[redunRel[i - 1].h] = i - 1;
			redunlefRel[redunRel[i].h] = i;
		}
	}
	redunlefHead[redunHead[0].h] = 0;
	redunrigHead[redunHead[redunTotal - 1].h] = redunTotal - 1;
	redunlefTail[redunTail[0].t] = 0;
	redunrigTail[redunTail[redunTotal - 1].t] = redunTotal - 1;
	redunlefRel[redunRel[0].h] = 0;
	redunrigRel[redunRel[redunTotal - 1].h] = redunTotal - 1;

	redunleft_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	redunright_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = redunlefHead[i] + 1; j <= redunrigHead[i]; j++)
			if (redunHead[j].r != redunHead[j - 1].r)
				redunleft_mean[redunHead[j].r] += 1.0;
		if (redunlefHead[i] <= redunrigHead[i])
			redunleft_mean[redunHead[redunlefHead[i]].r] += 1.0;
		for (INT j = redunlefTail[i] + 1; j <= redunrigTail[i]; j++)
			if (redunTail[j].r != redunTail[j - 1].r)
				redunright_mean[redunTail[j].r] += 1.0;
		if (redunlefTail[i] <= redunrigTail[i])
			redunright_mean[redunTail[redunlefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		redunleft_mean[i] = redunfreqRel[i] / redunleft_mean[i];
		redunright_mean[i] = redunfreqRel[i] / redunright_mean[i];
	}


}

Triple *testList;
Triple *validList;
Triple *tripleList;

extern "C"
void importTestFiles() {
    FILE *fin;
    INT tmp;

    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);

    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);

    FILE* f_kb1, * f_kb2, * f_kb3;
    if (train_file == "")
        f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        f_kb2 = fopen(train_file.c_str(), "r");
    if (test_file == "")
        f_kb1 = fopen((inPath + "test2id.txt").c_str(), "r");
    else
        f_kb1 = fopen(test_file.c_str(), "r");
    if (valid_file == "")
        f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
    else
        f_kb3 = fopen(valid_file.c_str(), "r");
    tmp = fscanf(f_kb1, "%ld", &testTotal);
    tmp = fscanf(f_kb2, "%ld", &trainTotal);
    tmp = fscanf(f_kb3, "%ld", &validTotal);
    tripleTotal = testTotal + trainTotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    validList = (Triple *)calloc(validTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &testList[i].h);
        tmp = fscanf(f_kb1, "%ld", &testList[i].t);
        tmp = fscanf(f_kb1, "%ld", &testList[i].r);
        tripleList[i] = testList[i];
    }
    for (INT i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].h);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].t);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].r);
    }
    for (INT i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].h);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].t);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].r);
        validList[i] = tripleList[i + testTotal + trainTotal];
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
    std::sort(testList, testList + testTotal, Triple::cmp_rel2);
    std::sort(validList, validList + validTotal, Triple::cmp_rel2);
//    printf("The total of test triples is %ld.\n", testTotal);
//    printf("The total of valid triples is %ld.\n", validTotal);

    testLef = (INT *)calloc(relationTotal, sizeof(INT));
    testRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(testLef, -1, sizeof(INT) * relationTotal);
    memset(testRig, -1, sizeof(INT) * relationTotal);
    for (INT i = 1; i < testTotal; i++) {
	if (testList[i].r != testList[i-1].r) {
	    testRig[testList[i-1].r] = i - 1;
	    testLef[testList[i].r] = i;
	}
    }
    testLef[testList[0].r] = 0;
    testRig[testList[testTotal - 1].r] = testTotal - 1;

    validLef = (INT *)calloc(relationTotal, sizeof(INT));
    validRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(validLef, -1, sizeof(INT)*relationTotal);
    memset(validRig, -1, sizeof(INT)*relationTotal);
    for (INT i = 1; i < validTotal; i++) {
	if (validList[i].r != validList[i-1].r) {
	    validRig[validList[i-1].r] = i - 1;
	    validLef[validList[i].r] = i;
	}
    }
    validLef[validList[0].r] = 0;
    validRig[validList[validTotal - 1].r] = validTotal - 1;
}

INT* head_lef;
INT* head_rig;
INT* tail_lef;
INT* tail_rig;
INT* head_type;
INT* tail_type;

extern "C"
void importTypeFiles() {

    head_lef = (INT *)calloc(relationTotal, sizeof(INT));
    head_rig = (INT *)calloc(relationTotal, sizeof(INT));
    tail_lef = (INT *)calloc(relationTotal, sizeof(INT));
    tail_rig = (INT *)calloc(relationTotal, sizeof(INT));
    INT total_lef = 0;
    INT total_rig = 0;
    FILE* f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    INT tmp;
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld %ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_lef++;
        }
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_rig++;
        }
    }
    fclose(f_type);
    head_type = (INT *)calloc(total_lef, sizeof(INT)); 
    tail_type = (INT *)calloc(total_rig, sizeof(INT));
    total_lef = 0;
    total_rig = 0;
    f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        head_lef[rel] = total_lef;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        std::sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}


#endif
