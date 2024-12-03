# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:26:05 2022

@author: beni-
"""

import pandas as pd
import numpy as np 

n_seed = 5

with open('results/BGN999.txt', 'r') as det_ :
    det = [line.strip().split('\t') for line in det_]    
det_.close()

det = np.array(det)

df = pd.DataFrame(det[1:], columns=det[0])
df = df[['algo', 'dataset', 'seed', 'nb_neur', 'nb_hdlr',
         'nb_repl', 'train_loss', 'valid_loss', 'test_loss']]
df = df.sort_values(by = ['algo', 'dataset', 'seed', 'nb_neur', 'nb_hdlr',
         'nb_repl', 'train_loss', 'valid_loss', 'test_loss'])

df = df.astype({'train_loss': np.float})
df = df.astype({'valid_loss': np.float})
df = df.astype({'test_loss': np.float})
df = df.astype({'seed': np.float})
df = df.astype({'nb_neur': np.float})

df_1 = df.groupby(['algo', 'dataset', 'nb_hdlr', 'nb_repl', 'valid_loss']).min()
df_1 = df_1.reset_index()
df_2 = df_1[['algo', 'dataset', 'seed', 'train_loss', 'valid_loss', 'test_loss']].groupby(['algo', 'dataset', 'seed']).min()
df_2 = df_2.reset_index()
df_2 = df_2[['algo', 'dataset', 'seed', 'valid_loss']].merge(df[['algo', 'dataset', 'seed', 'train_loss', 'valid_loss', 'test_loss']],
                      left_on=['algo', 'dataset', 'valid_loss'],
                      right_on=['algo', 'dataset', 'valid_loss'])[['algo', 'dataset', 'train_loss', 'valid_loss', 'test_loss']]
df_2 = df_2.groupby(['algo', 'dataset', 'valid_loss']).mean()
df_2 = df_2.groupby(['algo', 'dataset']).std()
df_2 = df_2.reset_index()

df_1 = df_1[['algo', 'dataset', 'train_loss', 'test_loss', 'valid_loss']].groupby(['algo', 'dataset']).min()
df_1 = df_1.reset_index()[['algo', 'dataset', 'valid_loss']]
df_3 = df_1.merge(df, left_on=['algo', 'dataset', 'valid_loss'],
                      right_on=['algo', 'dataset', 'valid_loss'])[['algo', 'dataset', 'nb_neur', 'train_loss', 'valid_loss', 'test_loss']]
df_3 = df_3.groupby(['algo', 'dataset', 'valid_loss']).min().reset_index()

df_3['train_std'] = df_2['train_loss']
df_3['test_std'] = df_2['test_loss']
print(df_3[['algo', 'dataset', 'nb_neur', 'train_loss', 'train_std','valid_loss', 'test_loss', 'test_std']])

#for i in range(len(df_3['dataset'])) :
#    print('& '+str(df_3['algo'][i]).upper()+\
#          ' & 1'+\
#          ' & '+str(df_3['nb_neur'][i])+\
#          ' & '+str(round(df_3['train_loss'][i],2))+" $\pm$ "+str(round(df_3['train_std'][i],2))+\
#          ' & '+str(round(df_3['test_loss'][i],2))+" $\pm$ "+str(round(df_3['test_std'][i],2))+"\\\\")
    
#& EBP & 2 & 10 & 0.001 $\pm$ 0.001 & 0.021 $\pm$ 0.004\\

