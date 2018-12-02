#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 10:55:45 2018

@author: asabater
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

'''
Sales 1: Ganancias = volumen * precio
Sales 2: Beneficio = Ganancias - Devoluciones
Investment X: tipos de inversion

Predecir por Cluster / Brand / Month
'''


class suppress_stdout_stderr(object):
	'''
	A context manager for doing a "deep suppression" of stdout and stderr in
	Python, i.e. will suppress all print, even if the print originates in a
	compiled C/Fortran sub-function.
	   This will not suppress raised exceptions, since exceptions are printed
	to stderr just before a script exits, and after the context manager has
	exited (at least, I think that is why it lets exceptions through).

	'''
	def __init__(self):
		# Open a pair of null files
		self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
		# Save the actual stdout (1) and stderr (2) file descriptors.
		self.save_fds = (os.dup(1), os.dup(2))

	def __enter__(self):
		# Assign the null pointers to stdout and stderr.
		os.dup2(self.null_fds[0], 1)
		os.dup2(self.null_fds[1], 2)

	def __exit__(self, *_):
		# Re-assign the real stdout/stderr back to (1) and (2)
		os.dup2(self.save_fds[0], 1)
		os.dup2(self.save_fds[1], 2)
		# Close the null files
		os.close(self.null_fds[0])
		os.close(self.null_fds[1])
		


dir_data = './data/Data_Novartis_Datathon-Participants.xlsx'
dir_submission_template = './data/Data_Novartis_Datathon-Results_Challenge1_Template.csv'
dir_results = './processed_data/'

df = pd.read_excel(dir_data, skiprows=3)
df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]


# %%

dfsales2 = df.loc[df['Function'] == "Sales 2"]
dfsales2 = dfsales2[dfsales2.columns.drop(list(dfsales2.filter(regex='2018')))]
del dfsales2['Function']
del dfsales2[' Country']

#column_names = dfsales2.columns[3:]


# %%

from fbprophet import Prophet
from tqdm import tqdm
from joblib import Parallel, delayed

plot_forecasting = False

groups = dfsales2.groupby(['Cluster','Brand Group']).agg('sum')


def get_prophet_forecasting(group_name, group, cap=None):

	data = group.reset_index()
	data.columns = ['ds', 'y']
	

	data['y'][data.y <= 0] = 0.0

	# Remove first null rows
	first_row = 0
	for i in range(len(data)): 
		if data.y[i] == 0:
			first_row = i+1
		else:
			break
	
	
	data = data.loc[first_row:, :]
	data['y'][data.y > 0] = np.log(data['y'][data.y > 0])
	
	if len(data) == 0:
#		frcst = forecastings[-1][1]
#		frcst['yhat'] = 0
		return [group_name[0], group_name[1], np.zeros(12)], [None, None]

	
	with suppress_stdout_stderr():
		
		try:
			m = Prophet(growth='logistic', weekly_seasonality=False, daily_seasonality=False)
			m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
			
			cap = max(data.y)
			data['cap'] = cap
			m.fit(data)
	 
			future = m.make_future_dataframe(periods=1*12, freq='M')
			future['cap'] = cap
			frcst = m.predict(future);
			
		except:
			return [group_name[0], group_name[1], np.zeros(12)], [None, None]

#		for field in ['yhat', 'yhat_lower', 'yhat_upper']: 
#			print('---- ',list(frcst[frcst[field] > 0][field]))
#			print(sum(frcst[field] > 0))
#			frcst.loc[frcst[field] > 0][field] = np.exp(list(frcst.loc[frcst[field] > 0][field].values))
#			print('*****', list(frcst[frcst[field] > 0][field]))
#			print(np.exp([2.02]))
#		
		
	return [group_name[0], group_name[1], np.exp(frcst['yhat'].values)], [m, frcst]


# %%

if False:
	
	# %%
	
	clust = 1
	bg = 17
	
	group_name = ('Cluster {}'.format(clust), 'Brand Group {}'.format(bg))
	group = groups.loc[group_name]
	
	[_,_,res], [model, frcst] = get_prophet_forecasting(group_name, group)
	
	model.plot(frcst);
	plt.show()
	plt.plot(list(range(len(res[:-12]))), res[:-12])
	plt.plot(list(range(len(res[:-12]), len(res[:-12])+len(res[-12:]))), res[-12:])

	
# %%
	
forecasts = Parallel(n_jobs=-1)(delayed(get_prophet_forecasting)(group_name, group) 
					for group_name, group in tqdm(groups.iterrows(), total=len(groups)))


# %%

#forecasts, _ = forecasts
final_res = [ r[0] for r in forecasts ]
final_res = pd.DataFrame(final_res, columns=['Cluster', 'Brand Group', 'res'])


# %%

if False:
	
	# %%
	
	for group_name, group in tqdm(groups.iterrows(), total=len(groups)):
		print(group_name)
		forecasts, model = get_prophet_forecasting(group_name, group) 


# %%

submission_template = pd.read_csv(dir_submission_template)


bg_set = set()

#a = pd.DataFrame(final_res, columns=['Cluster', 'Brand Group', 'res'])

for i, r in submission_template.iterrows():
	bg = r['Brand Group'][12:].replace(',', '').split(' ')
	bg_set.update(bg)
	bg = [ 'Brand Group ' + str(i) for i in bg ]
	print(r['Cluster'], r['Brand Group'], bg)
	
#	print(type(r['res']))
	
	res = final_res[final_res['Cluster'] == r['Cluster']]
	res = [ rs['res'][-12:] for j, rs in res.iterrows() if rs['Brand Group'] in bg ]
#	res = res['res']
	res = np.sum(res, axis=0)
	submission_template.iloc[i, 2:] = res
	
	
for i, r in submission_template[submission_template['Brand Group'] == 'others'].iterrows():
	print(r.Cluster)
	
	res = final_res[final_res['Cluster'] == r['Cluster']]
	res1 = [ rs['res'][-12:] for j, rs in res.iterrows() if rs['Brand Group'][12:] not in bg_set ]

	res2 = np.sum(res1, axis=0)
	submission_template.iloc[i, 2:] = res2
	
#	if r.Cluster == 'Cluster 1': break


# %%

#for col in submission_template.columns[2:]:
#	submission_template.loc[:, col] = submission_template[col].apply(lambda x: x if x < 999999 else 0.0)



# %%



submission_template.to_csv(dir_results + str(len(os.listdir(dir_results))) + '_team15_prft_0.csv', index=False)
	
