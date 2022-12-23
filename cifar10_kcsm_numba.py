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
            for i1 in range(subdim):
                for j2 in range(subdim):
                    for s1 in range(class_info[c1]):
                        for t2 in range(class_info[c2]):
                            K_d[subdim * c1 + i1, subdim * c2 + j2] += K_cl[class_index[c1] + s1, i1] * K_cl[class_index[c2] + t2, j2] * K_all[class_index[c1] + s1, class_index[c2] + t2]

@jit
def gauss_projection_diff(x, K_p):
    for i_data in range(x.shape[0]):
        for i_gds in range(b.shape[1]):
            for c1 in range(class_num):
                for i1 in range(subdim):
                    for s1 in range(class_info[c1]):
                        K_p[i_gds, i_data] += b[i_gds, subdim * c1 + i1] * K_cl[class_index[c1] + s1, i1] * x[i_data, class_index[c1] + s1]

@jit
def gauss_gram_two(X, Y, K_t):
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            for k in range(X.shape[1]):
                b = (X[i][k] - Y[j][k])
                K_t[i][j] += b * b

def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))


#loading data, Please modify "loading data" to match the format of your data!-----------
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
sgm = 0.1                                   #Please write the sigma value.
subdim = 10                                 #Please write the dimensions of class subspace.
gdsdim = 20                                 #Please write the dimensions of GDS.                           
class_num = 3                               #Please write the number of class.
class_info = np.array([1000, 2000, 1000])   #Please write the number of data in each class.
#----------------------------------------------------------------------------------------


#normalization---------------------------------------------------------------------------
class_index = []
count_c = 0
for class_i in class_info:
    if count_c == 0:
        class_index.append(0)
        class_index.append(class_info[0])
    else:
        class_index.append(class_index[count_c] + class_info[count_c])
    count_c += 1
class_index = np.array(class_index)

count = 0
for train_data in train_x:
    train_x[count] = normalize(train_data, axis=1)
    count += 1
count = 0
for test_data in test_x:
    test_x[count] = normalize(test_data, axis=1)
    count += 1
#----------------------------------------------------------------------------------------


#kernel for each class-------------------------------------------------------------------
K_class = []
for train_data in train_x:
    K = np.zeros((train_data.shape[0], train_data.shape[0]))
    gauss_gram_mat(train_data, K)
    K = np.exp(- K / (2 * sgm))
    w, v = np.linalg.eigh(K)
    w, v = w[::-1], v[:, ::-1]
    rank = np.linalg.matrix_rank(K)
    w, v = w[:rank], v[:, :rank]
    d = v[:, 0:subdim]
    K_class.append(d)
#-----------------------------------------------------------------------------------------

#kernel for generalized difference subspaces----------------------------------------------
count_k = 0
for i in range(class_num):
    if i == 0:
        K_cl = K_class[i]
    else:
        K_cl = np.concatenate([K_cl, K_class[i]])

K_D = np.zeros((subdim * class_num, subdim * class_num))

for i in range(class_num):
    if i == 0:
        train_all = train_x[i]
    else:
        train_all = np.concatenate([train_all, train_x[i]])

K_all = np.zeros((train_all.shape[0], train_all.shape[0]))
                                            
gauss_gram_mat(train_all, K_all)
K_all = np.exp(- K_all / (2 * sgm))

gauss_class_mat_diff(K_D)
B, b = np.linalg.eigh(K_D)
b = b[:, 0:gdsdim]
print("training complete")
#-----------------------------------------------------------------------------------------


#project data onto gds--------------------------------------------------------------------
test_np = np.array(test_x)
test_np = test_np.reshape([test_np.shape[0], test_np.shape[2]])

train_data_projected = np.zeros((gdsdim, train_all.shape[0]))
gauss_projection_diff(K_all, train_data_projected)
print("projection complete 1 (train(class) data)")

K_two = np.zeros((test_np.shape[0], train_all.shape[0]))
gauss_gram_two(test_np, train_all, K_two)
K_two = np.exp(- K_two / (2 * sgm))

test_data_projected = np.zeros((gdsdim, test_np.shape[0]))
gauss_projection_diff(K_two, test_data_projected)
print("projection complete 2 (test data)")
#-----------------------------------------------------------------------------------------


#generate linear subspace for train data projected onto gds-------------------------------
subspace_class = []
for i in range(class_num):
    data_projected = train_data_projected[:, class_index[i]:class_index[i+1]]
    w, v = np.linalg.eigh(data_projected @ data_projected.T)
    w, v = w[::-1], v[:, ::-1]
    rank = np.linalg.matrix_rank(K)
    w, v = w[:rank], v[:, :rank]
    d = v[:, 0:subdim]
    subspace_class.append(d)
#-----------------------------------------------------------------------------------------


#calculate projection length--------------------------------------------------------------
similarity_all = []
for i in range(test_np.shape[0]):
    data_input = test_data_projected[:, i]
    similarity_one = []
    for subspace_class_i in subspace_class:
        length = np.linalg.norm(subspace_class_i.T @ data_input, ord = 2)
        similarity_one.append(length)
    similarity_all.append(np.argmax(np.array(similarity_one))) 
#------------------------------------------------------------------------------------------

print("accuracy:", accuracy_score(similarity_all, test_y))