#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 17:37:30 2018

@author: asabater
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# TODO: enrtenar y predecir con variables laged
# TODO: Añadir diferencias
# TODO: Añadir trend
# TODO: Agrupar por submission antes de sacar features


dir_data = './data/Data_Novartis_Datathon-Participants.xlsx'
dir_submission_template = './data/Data_Novartis_Datathon-Results_Challenge1_Template.csv'
dir_results = './processed_data/'

df = pd.read_excel(dir_data, skiprows=3)
df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]

df = df.rename(columns={' Country': 'Country'})
#df = df[df.columns.drop(list(df.filter(regex='2018')))]


# %%

n_lags = 12

lags = list(range(0, n_lags))

groups = df.groupby(by=['Cluster', 'Brand Group'])
dataset = []

for group_info, g in tqdm(groups, total=len(groups)):
	
	g = g.groupby(by='Function').agg('sum').reset_index()
	
	cluster, bg = group_info
	
#	del g['Country']
	
	if 'Sales 2' not in g['Function'].tolist(): continue
	
	sales_2 = g[g['Function'] == 'Sales 2'].iloc[:,1:].values.tolist()[0]

	if 'Sales 1' in g['Function'].tolist(): 
		sales_1 = g[g['Function'] == 'Sales 1'].iloc[:,1:].values.tolist()[0]
	else:
		sales_1 = [0.0]*len(sales_2)
		

#	months = { 'month_{}'.format(m): [0]*len(sales_2) for m in range(1,13) }
	months = {}
	
	data = {**{
#			'sales_1': sales_1,
#			'sales_2': sales_2,
			'y': sales_2[1:] + [np.nan],
			}, 
			**months}
	
	for inv in range(1, 6):
		col = 'Investment {}'.format(inv)
		if col in g['Function'].tolist(): data[col] = g[g['Function'] == col].iloc[:,1:].values.tolist()[0]
		else: data[col] = [0.0]*len(sales_2)
	
	for l in lags:
		data['sales_1-{}'.format((l+1))] = [0]*l + sales_1[:len(sales_2)-l]
		data['sales_2-{}'.format((l+1))] = [0]*l + sales_2[:len(sales_2)-l]

		for inv in range(1, 6):
			col = 'Investment {}'.format(inv)
			data[col+'-{}'.format((l+1))] = [0]*l + data[col][:len(sales_2)-l]
			
	
	df_group = pd.DataFrame(data, index = pd.to_datetime(g.columns[1:]))
	df_group['month'] = df_group.index.month.tolist()

	
#	df_group_columns = list(df_group.columns)
#	for i,m in enumerate(df_group.index.month):
##		df_group.iloc[i, df_group_columns.index('month_{}'.format(m))] = 1
#		df_group.iloc[i, df_group_columns.index('month_{}'.format(m))] = 1
	
	
	dataset.append({
				'Cluster': group_info[0],
				'Brand Group': group_info[1],
#				'Country': group_info[2],
				'data': df_group,
			})
	
#	break


# %%
	
import pickle

dir_dataset = './datasets/laged_by_cluster_brand_nocountry_l{}.pkl'.format(n_lags)
print(dir_dataset)
pickle.dump(dataset, open(dir_dataset, 'wb'))

