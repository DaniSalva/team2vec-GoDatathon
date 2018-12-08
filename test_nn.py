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


dataset = pickle.load(open('./datasets/laged_by_cluster_brand_country_l12.pkl', 'rb'))


# %%

X = []
y = []
scalersX = []
scalersY = []
columns = dataset[0]['data'].columns.tolist()
#good_columns = [ c for c in columns if 'sales_1' not in c ]
#good_columns = [ c for c in good_columns if 'Investment' not in c ]


good_columns = [ c for c in columns if 'sales_1' not in c ]
#good_columns = [ c for c in good_columns if 'month_' not in c ]
#good_columns = [ c for c in good_columns if 'Investment' not in c ]


for d in dataset:
	data = d['data'].loc[:,good_columns]
	data = data[data.index < '2017-12-01 00:00:00']
#	data = data[np.isfinite(data['y'])]
#	data = data[data['sales_2-12'] != 0]
#	data = data[data['sales_2-{}'.format(max([ int(c.split('-')[-1]) for c in data.columns if c.startswith('sales_2-') ]))] != 0]
	
	d['data'] = data
	
	y_d = data['y'].values.reshape(-1, 1)
	scl_y_d = StandardScaler()
	y_d = scl_y_d.fit_transform(y_d)
	
	y.append(y_d)
	del data['y']


	X_d = data.values
	scl_X_d = StandardScaler()
	X_d = scl_X_d.fit_transform(X_d)
	
	X.append(X_d)
	
	scalersX.append(scl_X_d)
	scalersY.append(scl_y_d)
	

X = np.vstack(X)
y = np.vstack(y)



#X, y = shuffle(X, y, random_state=0)


# %%
	
# =============================================================================
# NN
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger, ReduceLROnPlateau


model = Sequential()
model.add(Dense(128, input_dim=(X.shape[1])))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(64, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


callbacks = [
			ModelCheckpoint('./nn_weights.h5', 
				   monitor='val_loss', verbose=1, save_best_only=True),
				   
			EarlyStopping(monitor='val_loss', min_delta=0.00001, 
				 verbose=1, patience=20),
		]

model.fit(X, y, validation_split=0.25, epochs=1000, callbacks = callbacks, batch_size = 1024, shuffle=True)

model.load_weights('./nn_weights.h5')

models = [model]


# %%

#from sklearn.ensemble import GradientBoostingRegressor
#
#model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1,
#			max_depth=5, random_state=0, loss='ls').fit(X_train, y_train)


# %%

dataset = pickle.load(open('./datasets/laged_by_cluster_brand_nocountry_l12.pkl', 'rb'))

for ind_scl, d in tqdm(enumerate(dataset), total=len(dataset)):

	
#	d = dataset[0]

	data = d['data'].loc[:,good_columns]
#	data = data[~np.isfinite(data['y'])]
	data = data[data.index >= '2017-12-01 00:00:00']
	
	good_columns_pred = good_columns.copy()
	del good_columns_pred[good_columns_pred.index('y')]
	del data['y']
	preds = []
	
	for i, (ind, r) in enumerate(data.iterrows()):
		
#		print('*****', i, len(data))
		if i > len(data)-1: break
		
#		pred = np.inf
#		pred = model.predict(data.iloc[i])	
	
		pred_data = data.loc[ind].values
		pred_data = scalersX[ind_scl].transform(pred_data.reshape(1, X.shape[1]))
		pred = [ m.predict(pred_data)[0][0] for m in models ]
		pred = np.mean(pred)
		preds.append(pred)
#		break
	
#		print('***', i+1)
#		if i < len(data): data.iloc[i+1, good_columns_pred.index('sales_2-1')] = pred
		
		for l in range(1, 13):
#			print(i+l+1, 'sales_2-{}'.format(l))
#			print('---', i+l)
			if i+l < len(data): 
#				data.iloc[i+l, good_columns_pred.index('sales_2-{}'.format(l))] = '{} - {}'.format(i,l)
				data.iloc[i+l, good_columns_pred.index('sales_2-{}'.format(l))] = pred
				
#	preds = data.sales_2[-12:].tolist()
	preds = scalersY[ind_scl].inverse_transform(preds[1:])
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
	res = np.sum(res, axis=0)
	
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

filename = dir_results + str(len(os.listdir(dir_results))) + '_team15_dt_{}_v2.csv'.format('XGB')
print(filename)
submission_template.to_csv(filename, index=False)



