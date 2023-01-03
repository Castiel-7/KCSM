# %%
import os
import numpy as np
from numpy.random import randint, rand
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import hdf5storage
import random

import pandas as pd
from numpy.random import randint, rand
from numba import jit, void, f8, njit
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import eigh

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer


def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))

def l2_kernel(X, Y):
    XX = (X ** 2).sum(axis=0)[:, None]
    YY = (Y ** 2).sum(axis=0)[None, :]

    return XX - 2 * (X.T @ Y) + YY

def rbf_kernel(X, Y, sigma=None):
    n_dims = X.shape[0]
    if sigma is None:
        sigma = np.sqrt(n_dims / 2)

    x = l2_kernel(X, Y)

    return np.exp(-0.5 * x / (sigma ** 2))

def dual_vectors(K, n_subdims=None, higher=True, eps=1e-6):
    """
    Calc dual representation of vectors in kernel space

    Parameters:
    -----------
    K :  array-like, shape: (n_samples, n_samples)
        Grammian Matrix of X: K(X, X)
    n_subdims: int, default=None
        Number of vectors of dual vectors to return
    higher: boolean, default=None
        If True, this function returns eigenbasis corresponding to 
            higher `n_subdims` eigenvalues in descending order.
        If False, this function returns eigenbasis corresponding to 
            lower `n_subdims` eigenvalues in descending order.
    eps: float, default=1e-20
        lower limit of eigenvalues

    Returns:
    --------
    A : array-like, shape: (n_samples, n_samples)
        Dual replesentation vectors.
        it satisfies lambda[i] * A[i] @ A[i] == 1, where lambda[i] is i-th biggest eigenvalue
    e:  array-like, shape: (n_samples, )
        Eigen values descending sorted
    """

    eigvals = _get_eigvals(K.shape[0], n_subdims, higher)
    e, A = _eigen_basis(K, eigvals=eigvals)

    # replace if there are too small eigenvalues
    e[(e < eps)] = eps

    A = A / np.sqrt(e)

    return A, e

def _prepare_X(X):
        """
        preprocessing data matricies X.
        normalize and transpose

        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_samples, n_dims)
        """

        # transpose to make feature vectors as column vectors
        # this makes it easy to implement refering to formula
        X = [_X.T for _X in X]

        return X

def _eigh(X, eigvals=None):
    """
    A wrapper function of numpy.linalg.eigh and scipy.linalg.eigh

    Parameters
    ----------
    X: array-like, shape (a, a)
        target symmetric matrix
    eigvals: tuple, (lo, hi)
        Indexes of the smallest and largest (in ascending order) eigenvalues and corresponding eigenvectors
        to be returned: 0 <= lo <= hi <= M-1. If omitted, all eigenvalues and eigenvectors are returned.

    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        eigenvalues with descending order
    V: array-like, shape (a, a) or (a, n_dims)
        eigenvectors
    """

    if eigvals != None:
        e, V = eigh(X, eigvals=eigvals)
    else:
        # numpy's eigh is faster than scipy's when all calculating eigenvalues and eigenvectors
        e, V = np.linalg.eigh(X)

    e, V = e[::-1], V[:, ::-1]

    return e, V

def _eigen_basis(X, eigvals=None):
    """
    Return subspace basis using PCA

    Parameters
    ----------
    X: array-like, shape (a, a)
        target matrix
    n_dims: integer
        number of basis

    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        eigenvalues with descending order
    V: array-like, shape (a, a) or (a, n_dims)
        eigenvectors
    """

    try:
        e, V = _eigh(X, eigvals=eigvals)
    except np.linalg.LinAlgError:
        # if it not converges, try with tiny salt
        salt = 1e-8 * np.eye(X.shape[0])
        e, V = eigh(X + salt, eigvals=eigvals)

    return e, V

def _get_eigvals(n, n_subdims, higher):
    """
    Culculate eigvals for eigh
    
    Parameters
    ----------
    n: int
    n_subdims: int, dimension of subspace
    higher: boolean, if True, use higher `n_subdim` basis

    Returns
    -------
    eigvals: tuple of 2 integers
    """

    if n_subdims is None:
        return None

    if higher:
        low = max(0, n - n_subdims)
        high = n - 1
    else:
        low = 0
        high = min(n - 1, n_subdims - 1)

    return low, high

def subspace_bases(X, n_subdims=None, higher=True, return_eigvals=False):
    """
    Return subspace basis using PCA

    Parameters
    ----------
    X: array-like, shape (n_dimensions, n_vectors)
        data matrix
    n_subdims: integer
        number of subspace dimension
    higher: bool
        if True, this function returns eigenvectors collesponding
        top-`n_subdims` eigenvalues. default is True.
    return_eigvals: bool
        if True, this function also returns eigenvalues.
    Returns
    -------
    V: array-like, shape (n_dimensions, n_subdims)
        bases matrix
    w: array-like shape (n_subdims)
        eigenvalues
    """

    if X.shape[0] <= X.shape[1]:
        eigvals = _get_eigvals(X.shape[0], n_subdims, higher)
        # get eigenvectors of autocorrelation matrix X @ X.T
        w, V = _eigen_basis(X @ X.T, eigvals=eigvals)
    else:
        # use linear kernel to get eigenvectors
        A, w = dual_vectors(X.T @ X, n_subdims=n_subdims, higher=higher)
        V = X @ A

    if return_eigvals:
        return V, w
    else:
        return V

# @jit
def gauss_gram_two(X, Y, K_t):
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            for k in range(X.shape[0]):
                b = (X[i][k] - Y[j][k])
                K_t[i][j] += b * b

def _gds_projection(gds, bases):
        """
        GDS projection.
        Projected bases will be normalized and orthogonalized.

        Parameters
        ----------
        bases: arrays, (n_dims, n_subdims)

        Returns
        -------
        bases: arrays, (n_gds_dims, n_subdims)
        """

        # bases_proj, (n_gds_dims, n_subdims)
        bases_proj = np.matmul(gds.T, bases)
        qr = np.vectorize(np.linalg.qr, signature="(n,m)->(n,m),(m,m)")
        bases, _ = qr(bases_proj)
        return bases

def _gds_projection_in(gds, bases):

        # bases_proj, (n_gds_dims, n_subdims)
        bases_proj = np.matmul(gds.T, bases[0])
        qr = np.vectorize(np.linalg.qr, signature="(n,m)->(n,m),(m,m)")
        bases, _ = qr(bases_proj)
        return bases

def z(X, w, b, m):
    #print(w.shape, X.shape, b.shape)
    return np.sqrt(2 / m) * np.cos((w @ X).T + b).T

def get_newX(X):
       
    n_gds_dims = 11 #if you use small subspaces, take "11", if big subspaces, take "18"
    n_subdims = 6
    dic = None
    dic = []

    m = 100
    sigma = 0.1

    n_dims = X[0].shape[0]
    if sigma is None:
        sigma = np.sqrt(n_dims / 2)

    

    newX = [] #here we create the newX vector (m-dimensional) from the dim of reference subspaces using rff approximation (mathematically similar), cuz it's faster! check self.w and self.b 
    for _X in X: 
        _newX = []
        for n in range(n_approx):
            _newX.append(z(_X, w7[n], b7[n], m))
        _newX = np.stack(_newX, axis=0).mean(axis=0)
        newX.append(_newX)

    newX_list_of_list = []
    for x in newX:
        newX_list_of_list.append([x])

    for newXx in newX_list_of_list:
        newXx = np.array(newXx)
        dic_ = [subspace_bases(_X, n_subdims) for _X in newXx] #PCA for the NEW reference subspaces (Uchiyama san code)
        dic.append(dic_)
        
    #mn hna jdid te3 rff_csm
    dic = np.array(dic)
    dic_ = dic[:, 0, :, :] #hna na7it l7kaya te3 list of list puisk rah ndiro gds

    all_bases = np.hstack((dic_[0], dic_[1])) #hna dert stack le list 0 w list 1
      
    # n_gds_dims
    if 0.0 < n_gds_dims <= 1.0:
        n_gds_dims = int(all_bases.shape[1] * n_gds_dims)
    else:
        n_gds_dims = n_gds_dims

    # gds, (n_dims, n_gds_dims)
    gds = subspace_bases(all_bases, n_gds_dims, higher=False)

    dic = _gds_projection(gds,dic_)

    return dic,gds,all_bases
# #----------------------------------------------------------------------------------------





X_safe = hdf5storage.loadmat('X_safe.mat')
X_mal = hdf5storage.loadmat('X_mal.mat')
in_mal = hdf5storage.loadmat('in_mal.mat')

# we did transpose because matlab take the first dimension of an array as Dim(HxW)
# but here, the first dimension is the number of samples!
X_safe = np.array(X_safe['X_safe']).T
X_mal = np.array(X_mal['X_mal']).T
in_mal = np.array(in_mal['in_mal']).T

# Calculate the number of elements to choose from each array
num_X_safe = int(X_safe.shape[0] * 0.6)  #take rand 60% for safe train
num_X_mal = int(X_mal.shape[0] * 0.2)    #take rand 20% for mal train
num_in_mal = int(in_mal.shape[0] * 0.5)  #take rand 50% for mal input

X_safe = X_safe[np.random.permutation(len(X_safe))]
X_mal = X_mal[np.random.permutation(len(X_mal))]
in_mal = in_mal[np.random.permutation(len(in_mal))]

X_safe_chosen = X_safe[:num_X_safe]
X_mal_chosen = X_mal[:num_X_mal]
in_mal_chosen = in_mal[:num_in_mal]


test_x = np.concatenate([X_safe[num_X_safe:], in_mal_chosen])
test_x = [test_x[i, np.newaxis, :] for i in range(test_x.shape[0])]
test_y0 = np.full(len(X_safe[num_X_safe:]), 0)
test_y1 = np.full(len(in_mal_chosen), 1)
test_y = np.concatenate([test_y0, test_y1]) 
print(test_y)



#parameter-------------------------------------------------------------------------------                                 
class_num = 2                               
class_info = np.array([366, 589])   

#------------------------djo rff_CSM------------------------
#------------------------for training------------------------

train_x = [X_safe_chosen, X_mal_chosen]  
train_y = np.array([0, 1]) 
print(train_x[0].shape)

n_dims = train_x[0].shape[1]
print(n_dims)
sigma = 0.1                                          #parameter
n_approx = 1
m = 100                                              #parameter!(number of random numbers to generate)
w = []
b = []
for n in range(n_approx):
    w.append(np.random.randn(m, n_dims) / sigma)
    b.append(np.random.rand(m) * 2 * np.pi)
w7 = np.stack(w)
b7 = np.stack(b)
print(w7.shape, b7.shape)


# #normalization---------------------------------------------------------------------------
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


# preprocessing data matricies (transpose the shape of matrices)
train_x = _prepare_X(train_x)


#----------------------------------------------------------------------------------------


# rff for training data + PCA 
New_X,gds,all_bases = get_newX(train_x)
print(gds.shape)
print(len(New_X))
print(New_X.shape)
#print(test_x[0].shape)

#calculate similarity 
similarity_all = []
for i in range(len(test_x)):
    data_input = test_x[i]
    #print(w7.shape, b7.shape)
    z_in = z(data_input.T, w7[0,:], b7[0, :], m)
    z_projected = gds.T @ z_in
    #print(z_projected.shape)
    #print(data_input.shape)
    #print(z_in.shape)
    similarity_one = []
    for subspace_class_i in New_X:
        print(subspace_class_i.shape, z_projected.shape)
        length = np.linalg.norm(subspace_class_i.T @ z_projected, ord = 2)
        print(length)
        similarity_one.append(length)
    similarity_all.append(np.argmax(np.array(similarity_one))) 


#---------------------------------------------------------------


print("accuracy:", accuracy_score(similarity_all, test_y))