#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:41:53 2018

@author: asabater
"""

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


dir_dataset = './datasets/laged_by_cluster_brand_country_l12.pkl'

dataset = pickle.load(open(dir_dataset, 'rb'))



X = []
y = []
scalersX = []
scalersY = []
columns = dataset[0]['data'].columns.tolist()
#good_columns = [ c for c in columns if 'sales_1' not in c ]
#good_columns = [ c for c in good_columns if 'Investment' not in c ]


good_columns = [ c for c in columns if 'sales_1' not in c ]
good_columns = [ c for c in good_columns if 'month_' not in c ]
good_columns = [ c for c in good_columns if 'Investment' not in c ]


for d in dataset:
	data = d['data'].loc[:,good_columns]
	data = data[np.isfinite(data['y'])]
#	data = data[data['sales_2-12'] != 0]
#	data = data[data['sales_2-{}'.format(max([ int(c.split('-')[-1]) for c in data.columns if c.startswith('sales_2-') ]))] != 0]
	
#	first_row = 0
#	for i, v in enumerate(data['sales_2-12']):
#		if v != 0:
#			first_row = i
#			break
##	print(first_row)
#	data = data.iloc[first_row:,:]
	
	d['data'] = data
	
	y_d = data['y'].values.reshape(-1, 1)
#	y_d[y_d>0] = np.log(y_d[y_d>0])
#	scl_y_d = StandardScaler()
#	y_d = scl_y_d.fit_transform(y_d)
	
	y.append(y_d)
	del data['y']
	
	
	X_d = data.values
#	X_d[X_d>0] = np.log(X_d[X_d>0])
#	scl_X_d = StandardScaler()
#	X_d = scl_X_d.fit_transform(X_d)
	
	X.append(data.values)

#	scalersX.append(scl_X_d)
#	scalersY.append(scl_y_d)
	

X = np.vstack(X)
y = np.vstack(y)


#scalerX = StandardScaler()
#X = scalerX.fit_transform(X)
#scalerY = StandardScaler()
#y = scalerY.fit_transform(y.reshape(-1,1))

#X, y = shuffle(X, y, random_state=0)


# %%

# =============================================================================
# GradientBoostingRegressor
# =============================================================================

#kf = KFold(n_splits=5)
#
#models = Parallel(n_jobs=-1)(delayed(GradientBoostingRegressor(
#				n_estimators=1200, learning_rate=0.1,
#				max_depth=12, random_state=0, loss='ls').fit)
#				(X[train_index], y[train_index]) for train_index, test_index in kf.split(X))


# %%

# =============================================================================
# SVR Regressor
# =============================================================================
	
#from sklearn.svm import SVR
#
#model_name = 'SVR'
#clf = SVR()
#clf.fit(X, y)
#
#models = [clf]


# %%
	
# =============================================================================
# XGBoost
# =============================================================================

#import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

model_name = 'XGB'
n_splits = 4
models = []

#kf = KFold(n_splits=n_splits)
#for train_index, test_index in kf.split(X):
#	
#	model = XGBRegressor(
#			max_depth=16,
#			n_estimators=1200,
#			min_child_weight=50,
#			colsample_bytree=0.8,
#			subsample=0.8,
#			eta=0.3,
#			seed=42,
#			n_jobs=-1)
#	model.fit(
#			X[train_index], 
#			y[train_index],
#			eval_metric = 'rmse',
#			eval_set=[(X[train_index], y[train_index]), (X[test_index], y[test_index])],
#			early_stopping_rounds = 30
#			)
#	
#	models.append(model)


models = []
scores = []
for i in range(4):
	X_train, X_test, y_train, y_test = train_test_split(
								X, y, test_size=0.25, shuffle=True)
	
	model = XGBRegressor(max_depth=16,
	   n_estimators=1200,
	   min_child_weight=50,
	   colsample_bytree=0.8,
	   subsample=0.8,
	   eta=0.3,
	   seed=42)
	
	
	model.fit(
	   X_train,
	   y_train,
	   eval_metric="rmse",
	   eval_set=[(X_train, y_train), (X_test, y_test)],
	   verbose=True,
	   early_stopping_rounds = 30)


	models.append(model)
	scores.append(model.best_score)
	


# %%

#from sklearn.ensemble import GradientBoostingRegressor
#
#model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,
#			max_depth=5, random_state=0, loss='ls').fit(X_train, y_train)


#models = [models[-1]]

# %%

dataset = pickle.load(open(dir_dataset, 'rb'))

for ind_scl, d in tqdm(enumerate(dataset), total=len(dataset)):

#	d = dataset[0]

	data = d['data'].loc[:,good_columns]
#	data = data[~np.isfinite(data['y'])]
	data = data[data.index >= '2017-12-01 00:00:00']

	good_columns_pred = good_columns.copy()
	del good_columns_pred[good_columns_pred.index('y')]
	del data['y']
	
#	preds = []
	
	for i, (ind, r) in enumerate(data.iterrows()):
		
		if i > len(data)-1: break
		
#		pred = np.inf
#		pred = model.predict(data.iloc[i])	
		sample = data.iloc[i].values.reshape(1, X.shape[1])
#		sample[sample>0] = np.log(sample[sample>0])
#		sample = np.log(sample)
#		sample = scalersX[ind_scl].transform(sample)
		pred = [ m.predict(sample)[0] for m in models ]
		pred = np.median(pred)
#		pred = scalersY[ind].inverse_transform(pred)
#		preds.append(pred)
	
#		print('***', i+1)
#		if i < len(data): data.iloc[i+1, good_columns_pred.index('sales_2')] = pred
		
		for l in range(1, 13):
#			print(i+l+1, 'sales_2-{}'.format(l))
			if i+l < len(data): 
#				data.iloc[i+l, good_columns_pred.index('sales_2-{}'.format(l))] = '{} - {}'.format(i,l)
				data.iloc[i+l, good_columns_pred.index('sales_2-{}'.format(l))] = pred
		
				
	preds = data['sales_2-1'][-12:].tolist()
#	preds = [ p  if p<=0 else np.exp(p) for p in preds ]
#	preds = np.exp(preds)
#	preds = scalersY[ind_scl].inverse_transform(preds)
	d['preds'] = preds


# %%

dir_submission_template = './data/Data_Novartis_Datathon-Results_Challenge1_Template.csv'

bg_set = set()
grouped_groups = []
submission_template = pd.read_csv(dir_submission_template)

for i,r in submission_template.iterrows():
	bg = r['Brand Group'][12:].replace(',', '').split(' ')
	bg_set.update(bg)
	bg = [ 'Brand Group ' + str(i) for i in bg ]
	print(r['Cluster'], r['Brand Group'], bg)
	
#	res = dfsales2[dfsales2['Cluster'] == r['Cluster']]
#	res = [ rs.iloc[2:].values for j, rs in res.iterrows() if rs['Brand Group'] in bg ]
	
	res = [ d['preds'] for d in dataset if d['Cluster'] == r.Cluster and d['Brand Group'] in bg]
	res = np.mean(res, axis=0)
	
	if r['Brand Group'] != 'others': 
#		grouped_groups.append((r.Cluster, r['Brand Group'], res))

		row = submission_template[(submission_template.Cluster==r.Cluster) & (submission_template['Brand Group'] == r['Brand Group'])]
		submission_template.iloc[row.index[0], 2:] = res[-12:]


for i, r in submission_template[submission_template['Brand Group'] == 'others'].iterrows():
#	res = dfsales2[dfsales2['Cluster'] == r['Cluster']]
#	res1 = [ rs.iloc[2:].values for j, rs in res.iterrows() if rs['Brand Group'][12:] not in bg_set ]
	res = [ d['preds'] for d in dataset if d['Cluster'] == r.Cluster and d['Brand Group'] not in bg_set ]
	res = np.sum(res, axis=0)
#	submission_template.iloc[i, 2:] = res2
#	grouped_groups.append((r.Cluster, 'others', res))
	
	row = submission_template[(submission_template.Cluster==r.Cluster) & (submission_template['Brand Group'] == 'others')]
	submission_template.iloc[row.index[0], 2:] = res[-12:]

grouped_groups = pd.DataFrame(grouped_groups, columns=['Cluster', 'Brand Group', 'res'])


# %%

import os
dir_results = './processed_data/'

filename = dir_results + str(len(os.listdir(dir_results))) + '_team15_dt_{}_v2.csv'.format(model_name)
print(filename)
submission_template.to_csv(filename, index=False)



