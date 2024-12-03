# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:26:05 2022

@author: beni-
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

with open('results/BGN_peptides_17.txt', 'r') as det_ :
    det = [line.strip().split('\t') for line in det_]    
det_.close()
det = np.array(det)
df = pd.DataFrame(det[1:], columns=det[0])
df = df[['algo', 'dataset', 'seed', 'fold_number', 'architecture', 'max_coef', 'train_loss', 'valid_loss', 'test_loss', 'R2']]
df = df.sort_values(by = ['algo', 'dataset', 'seed', 'fold_number', 'architecture', 'max_coef', 'train_loss', 'valid_loss', 'test_loss', 'R2'])

df = df.astype({'train_loss': float})
df = df.astype({'valid_loss': float})
df = df.astype({'test_loss': float})
df = df.astype({'seed': float})
df = df.astype({'fold_number': float})
df = df.astype({'architecture': float})
df = df.astype({'max_coef': float})
df = df.astype({'R2': float})
df[['train_loss']] = df[['train_loss']] ** 0.5
df[['valid_loss']] = df[['valid_loss']] ** 0.5
df[['test_loss']] = df[['test_loss']] ** 0.5

df_mea = df.groupby(['algo', 'dataset', 'architecture']).mean()
df_std = df.groupby(['algo', 'dataset', 'architecture']).std()
df_mea = df_mea.reset_index()
df_std = df_std.reset_index()
df_std = df_std.rename(columns={"test_loss": "test_std", "train_loss": "train_std", "R2": "R2_std"})
df = df_mea.merge(df_std, left_on=['algo', 'dataset', 'architecture'],
                         right_on=['algo', 'dataset', 'architecture'])
a_0 = np.array(df[  :15][['test_loss', 'test_std', 'R2', 'R2_std']])#[[1, 3, 5, 7, 9, 11]]
a_2 = np.array(df[15:30][['test_loss', 'test_std', 'R2', 'R2_std']])#[[1, 3, 5, 7, 9, 11]]
a_5 = np.array(df[30:45][['test_loss', 'test_std', 'R2', 'R2_std']])#[[1, 3, 5, 7, 9, 11]]
c_0 = np.array(df[45:60][['test_loss', 'test_std', 'R2', 'R2_std']])#[[1, 3, 5, 7, 9, 11]]
c_2 = np.array(df[60:75][['test_loss', 'test_std', 'R2', 'R2_std']])#[[1, 3, 5, 7, 9, 11]]
c_5 = np.array(df[75:  ][['test_loss', 'test_std', 'R2', 'R2_std']])#[[1, 3, 5, 7, 9, 11]]

data = np.array([c_0, c_2, c_5, a_0, a_2, a_5])
R2 = np.array([c_0[:,2], c_2[:,2], c_5[:,2], a_0[:,2], a_2[:,2], a_5[:,2]])
R2_tot = np.array([[0.36, 0.46, 0.51, 0.52, 0.55, 0.55, 0.55, 0.63, 0.63, 0.63, 0.63, 0.71, 0.71, 0.71, 0.71],
                   [0.35, 0.45, 0.47, 0.48, 0.51, 0.51, 0.61, 0.61, 0.61, 0.62, 0.62, 0.62, 0.62, 0.62, 0.64],
                   [0.35, 0.45, 0.46, 0.46, 0.46, 0.47, 0.47, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53, 0.53],
                   [0.44, 0.51, 0.54, 0.56, 0.57, 0.57, 0.57, 0.57, 0.57, 0.57, 0.58, 0.66, 0.66, 0.66, 0.66],
                   [0.44, 0.51, 0.53, 0.54, 0.54, 0.54, 0.59, 0.59, 0.59, 0.71, 0.91, 0.91, 0.91, 0.91, 0.92],
                   [0.44, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51]])

fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(R2_tot, cmap='Blues')

# Show all ticks and label them with the respective list entries
n_hyp = list(range(1,16))
exp_names = ['Cations', 'Cations 0.02', 'Cations 0.05', 'Anions', 'Anions 0.02', 'Anions 0.05']
ax.set_xticks(list(range(15)), n_hyp)
ax.set_yticks(list(range(6)), exp_names)

# Rotate the tick labels and set their alignment.
#plt.setp(ax.get_yticklabels(), rotation=90)

# Loop over data dimensions and create text annotations.
for i in range(len(n_hyp)):
    for j in range(len(exp_names)):
        col = 'black' if R2_tot[j, i] < 0.7 else 'white'
        text = ax.text(i, j, f'{round(data[j, i, 2],2)} ± {round(data[j, i, 3],2)} \n '
                             f'{round(data[j, i, 0],2)} ± {round(data[j, i, 1],2)} \n '
                             f'{R2_tot[j, i]}',
                               ha="center", va="center", color=col)

ax.set_title("Comparison of R^2 and RMSE for each experiments at each width")
fig.tight_layout()

from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)
plt.show()