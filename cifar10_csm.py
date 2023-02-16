import os
import numpy as np
from numpy.random import randint, rand
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from cvt.models import ConstrainedMSM
#from cvt.models import MutualSubspaceMethod

import pandas as pd
from numpy.random import randint, rand
from numba import jit, void, f8, njit
from sklearn.preprocessing import normalize

@jit(void(f8[:, :], f8[:, :]))
def gauss_gram_mat(x, K):
  n_points = len(x)
  n_dim = len(x[0])
  b = 0

  for j in range(n_points):
    for i in range(n_points):
      for k in range(n_dim):
        b = (x[i][k] - x[j][k])
        K[i][j] += b * b

@jit
def gauss_class_mat_diff(K_d):
    for c1 in range(class_num):
        for c2 in range(class_num):
            for i1 in range(subdim_info[c1]):
                for j2 in range(subdim_info[c2]):
                    for s1 in range(class_info[c1]):
                        for t2 in range(class_info[c2]):
                            K_d[subdim_index[c1] + i1, subdim_index[c2] + j2] += K_cl[class_index[c1] + s1, i1] * K_cl[class_index[c2] + t2, j2] * K_all[class_index[c1] + s1, class_index[c2] + t2]

@jit
def gauss_projection_diff(x, K_p):
    for i_data in range(x.shape[0]):
        for i_gds in range(b.shape[1]):
            for c1 in range(class_num):
                for i1 in range(subdim_info[c1]):
                    for s1 in range(class_info[c1]):
                        K_p[i_gds, i_data] += b[i_gds, subdim_index[c1] + i1] * K_cl[class_index[c1] + s1, i1] * x[i_data, class_index[c1] + s1]

@jit
def gauss_gram_two(X, Y, K_t):
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            for k in range(X.shape[1]):
                b = (X[i][k] - Y[j][k])
                K_t[i][j] += b * b

def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))


#loading data---Please modify "loading data" to match the format of your data!-----------
df_train = pd.read_csv(f'{parentpath1(__file__, f=0)}/cifar10_train.csv')
df_test = pd.read_csv(f'{parentpath1(__file__, f=0)}/cifar10_test2.csv')

df0 = df_train[df_train['label'] == 0]
df1 = df_train[df_train['label'] == 1]
df2 = df_train[df_train['label'] == 7]

train_x_0 = df0.loc[:, '1':'512'].values
train_x_1 = df1.loc[:, '1':'512'].values
train_x_2 = df2.loc[:, '1':'512'].values

dft0 = df_test[df_test['label'] == 0]
dft1 = df_test[df_test['label'] == 1]
dft2 = df_test[df_test['label'] == 7]

dft0_x = dft0.loc[:, '1':'512'].values
dft1_x = dft1.loc[:, '1':'512'].values
dft2_x = dft2.loc[:, '1':'512'].values

test_y0 = np.full(1000, 0)
test_y1 = np.full(1000, 1)
test_y2 = np.full(1000, 2)

#Please write train_x, train_y, test_x and test_y while following the comment rules below to match the format of your data!. 
train_x = [train_x_0[0:1000, :], train_x_1[0:2000, :], train_x_2[0:1000, :]]   #[(1000, 512), (2000, 512), (1000, 512)], I am listing 3 numpy arrays.(1000, 2000, 1000 is the number of data in classes 0, 1, 2)
                                                                               #(the number of data, dimensions(feature)) in list is numpy array. Please note that we group numpy arrays in lists.
train_y = np.array([0, 1, 2])                                                  #For example, for 3 classes, please write [0,1,2]
test_x = np.concatenate([dft0_x, dft1_x, dft2_x])
test_x = [test_x[i, np.newaxis, :] for i in range(test_x.shape[0])]            #(1. 512) : (the number of data, dimensions(feature)) in list(). Please note that we group numpy arrays in lists.
                                                                               #(1, 512) are lined up as many as the number of test data.
test_y = np.concatenate([test_y0, test_y1, test_y2])                           #Please create list from test data labels.
#In this code, the number of dimensions of the feature is set to 512, but the code works with any number of dimensions. 
# However, please write the number of classes and the number of data in each class in the "parameter" below.
#----------------------------------------------------------------------------------------

#parameter-------------------------------------------------------------------------------
gdsdim = 20                                  #Please write the dimensions of GDS.                           
subdim_info = np.array([10, 5, 10])          #You can choose the dimensions for class subspace!
#----------------------------------------------------------------------------------------

#create class subspace----------------------------------------------------------------------
count1 = 0
subspace_list = []
for train_x_i in train_x:
    train_self = train_x_i.T @ train_x_i
    w, v = np.linalg.eigh(train_self)
    w, v = w[::-1], v[:, ::-1]
    rank = np.linalg.matrix_rank(train_self)
    w, v = w[:rank], v[:, :rank]
    base = v[:, 0:subdim_info[count1]]
    subspace_list.append(base)
    count1 += 1
#------------------------------------------------------------------------------------------

#create gds--------------------------------------------------------------------------------
allbase = np.zeros((train_x[0].shape[1], train_x[0].shape[1]))
for subspace_i in subspace_list:
    allbase += subspace_i @ subspace_i.T
w, v = np.linalg.eigh(allbase)
w, v = w[::-1], v[:, ::-1]
rank = np.linalg.matrix_rank(allbase)
w, v = w[:rank], v[:, :rank]
gds = v[:, v.shape[1]-gdsdim:v.shape[1]]
#------------------------------------------------------------------------------------------

#project class subspace on gds-------------------------------------------------------------
subspace_ongds_list = []
for subspace_i in subspace_list:
    bases_proj = np.matmul(gds.T, subspace_i)
    qr = np.vectorize(np.linalg.qr, signature='(n,m)->(n,m),(m,m)')
    bases, _ = qr(bases_proj)
    subspace_ongds_list.append(bases)
#------------------------------------------------------------------------------------------

#calculate projection length--------------------------------------------------------------
similarity_all = []
for test_x_i in test_x:
    test_x_ongds = np.matmul(gds.T, test_x_i.T)
    #print(test_x_ongds.shape)
    similarity_one = []
    for subspace_ongds_i in subspace_ongds_list:
        length = np.linalg.norm(subspace_ongds_i.T @ test_x_ongds, ord = 2)
        similarity_one.append(length)
    similarity_all.append(np.argmax(np.array(similarity_one))) 
#------------------------------------------------------------------------------------------

print("accuracy:", accuracy_score(similarity_all, test_y))