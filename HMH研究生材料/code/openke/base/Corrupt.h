#ifndef CORRUPT_H
#define CORRUPT_H
#include "Random.h"
#include "Triple.h"
#include "Reader.h"

INT corrupt_head_train(INT id, INT h, INT r, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, entityTotal - 1);
		if (tmp < h)
			return tmp;
		else
			return tmp + 1;
	}
	lef = trainlefHead[h] - 1;
	rig = trainrigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = trainlefHead[h];
	rig = trainrigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

INT corrupt_tail_train(INT id, INT t, INT r, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, entityTotal - 1);
		if (tmp < t)
			return tmp;
		else
			return tmp + 1;
	}
	lef = trainlefTail[t] - 1;
	rig = trainrigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = trainlefTail[t];
	rig = trainrigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


INT corrupt_rel_train(INT id, INT h, INT t, INT r, bool p = false, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, relationTotal - 1);
		if (tmp < r)
			return tmp;
		else
			return tmp + 1;
	}
	lef = trainlefRel[h] - 1;
	rig = trainrigRel[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].t >= t) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = trainlefRel[h];
	rig = trainrigRel[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].t <= t) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp;
	if(p == false) {	
		tmp = rand_max(id, relationTotal - (rr - ll + 1));
	}
	else {
		INT start = r * (relationTotal - 1);
		REAL sum = 1;
		bool *record = (bool *)calloc(relationTotal - 1, sizeof(bool));
		for (INT i = ll; i <= rr; ++i){
			if (trainRel[i].r > r){
				sum -= prob[start + trainRel[i].r-1];
				record[trainRel[i].r-1] = true;
			}
			else if (trainRel[i].r < r){
				sum -= prob[start + trainRel[i].r];
				record[trainRel[i].r] = true;
			}
		}		
		REAL *prob_tmp = (REAL *)calloc(relationTotal-(rr-ll+1), sizeof(REAL));
		INT cnt = 0;
		REAL rec = 0;
		for (INT i = start; i < start + relationTotal - 1; ++i) {
			if (record[i-start])
				continue;
			rec += prob[i] / sum;
			prob_tmp[cnt++] = rec;
		}
		REAL m = rand_max(id, 10000) / 10000.0;
		lef = 0;
		rig = cnt - 1;
		while (lef < rig) {
			mid = (lef + rig) >> 1;
			if (prob_tmp[mid] < m) 
				lef = mid + 1; 
			else
				rig = mid;
		}
		tmp = rig;
		free(prob_tmp);
		free(record);
	}
	if (tmp < trainRel[ll].r) return tmp;
	if (tmp > trainRel[rr].r - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].r - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


bool _findtrain(INT h, INT t, INT r) {
    INT lef = 0;
    INT rig = tripleTotal - 1;
    INT mid;
    while (lef + 1 < rig) {
        INT mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
}


INT corrupt_head_redun(INT id, INT h, INT r, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, entityTotal - 1);
		if (tmp < h)
			return tmp;
		else
			return tmp + 1;
	}
	lef = redunlefHead[h] - 1;
	rig = redunrigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (redunHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = redunlefHead[h];
	rig = redunrigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (redunHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < redunHead[ll].t) return tmp;
	if (tmp > redunHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (redunHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

INT corrupt_tail_redun(INT id, INT t, INT r, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, entityTotal - 1);
		if (tmp < t)
			return tmp;
		else
			return tmp + 1;
	}
	lef = redunlefTail[t] - 1;
	rig = redunrigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (redunTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = redunlefTail[t];
	rig = redunrigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (redunTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < redunTail[ll].h) return tmp;
	if (tmp > redunTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (redunTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


INT corrupt_rel_redun(INT id, INT h, INT t, INT r, bool p = false, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, relationTotal - 1);
		if (tmp < r)
			return tmp;
		else
			return tmp + 1;
	}
	lef = redunlefRel[h] - 1;
	rig = redunrigRel[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (redunRel[mid].t >= t) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = redunlefRel[h];
	rig = redunrigRel[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (redunRel[mid].t <= t) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp;
	if(p == false) {
		tmp = rand_max(id, relationTotal - (rr - ll + 1));
	}
	else {
		INT start = r * (relationTotal - 1);
		REAL sum = 1;
		bool *record = (bool *)calloc(relationTotal - 1, sizeof(bool));
		for (INT i = ll; i <= rr; ++i){
			if (redunRel[i].r > r){
				sum -= prob[start + redunRel[i].r-1];
				record[redunRel[i].r-1] = true;
			}
			else if (redunRel[i].r < r){
				sum -= prob[start + redunRel[i].r];
				record[redunRel[i].r] = true;
			}
		}
		REAL *prob_tmp = (REAL *)calloc(relationTotal-(rr-ll+1), sizeof(REAL));
		INT cnt = 0;
		REAL rec = 0;
		for (INT i = start; i < start + relationTotal - 1; ++i) {
			if (record[i-start])
				continue;
			rec += prob[i] / sum;
			prob_tmp[cnt++] = rec;
		}
		REAL m = rand_max(id, 10000) / 10000.0;
		lef = 0;
		rig = cnt - 1;
		while (lef < rig) {
			mid = (lef + rig) >> 1;
			if (prob_tmp[mid] < m)
				lef = mid + 1;
			else
				rig = mid;
		}
		tmp = rig;
		free(prob_tmp);
		free(record);
	}
	if (tmp < redunRel[ll].r) return tmp;
	if (tmp > redunRel[rr].r - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (redunRel[mid].r - mid + ll - 1 < tmp)
			lef = mid;
		else
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


bool _findredun(INT h, INT t, INT r) {
    INT lef = 0;
    INT rig = tripleTotal - 1;
    INT mid;
    while (lef + 1 < rig) {
        INT mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
}



INT corrupt_train(INT h, INT r){
	INT ll = tail_lef[r];
	INT rr = tail_rig[r];
	INT loop = 0;
	INT t;
	while(true) {
		t = tail_type[rand(ll, rr)];
		if (not _findtrain(h, t, r)) {
			return t;
		} else {
			loop ++;
			if (loop >= 1000) {
				return corrupt_head_train(0, h, r);
			}
		} 
	}
}
INT corrupt_redun(INT h, INT r){
	INT ll = tail_lef[r];
	INT rr = tail_rig[r];
	INT loop = 0;
	INT t;
	while(true) {
		t = tail_type[rand(ll, rr)];
		if (not _findredun(h, t, r)) {
			return t;
		} else {
			loop ++;
			if (loop >= 1000) {
				return corrupt_head_redun(0, h, r);
			}
		}
	}
}
#endif
