#!/usr/bin/env python
# coding: utf-8

# In[7]:


from numpy import asarray as arr
from numpy import atleast_2d as twod
from numpy import asmatrix as mat
from numpy import ceil
from base import classifier
from utils import toIndex, fromIndex, to1ofK, from1ofK
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss


# In[3]:


class treeBase(object):

    def __init__(self, *args, **kwargs):
        self.L = arr([])           # indices of left children
        self.R = arr([])           # indices of right children
        self.F = arr([])           # feature to split on (-1 = leaf = predict)
        self.T = arr([])           # threshold to split on 
        self.P = arr([])           # prediction value for node
        self.sz = 0                 # size; also next node during construction
   
        if len(args) or len(kwargs):     # if we were given optional arguments,
            self.train(*args, **kwargs)    #  just pass them through to "train"
 
    
    def __repr__(self):
        to_return = 'Decision Tree\n'
        if len(self.T) > 8: return self.str_short()
        else:               return self.str_long()

    __str__ = __repr__

    def str_short(self): 
        ''' "Short" string representation of the decision tree (thresholds only)'''
        return 'Thresholds: {}'.format('[{0:.2f}, {1:.2f} ... {2:.2f}, {3:.2f}]'.format(self.T[0], self.T[1], self.T[-1], self.T[-2]));

    def str_long(self): 
        return self.__printTree(0,'  ');

    def train(self, X, Y, minParent=2, maxDepth=np.inf, minLeaf=1, nFeatures=None):
        n,d = mat(X).shape
        nFeatures = min(nFeatures,d) if nFeatures else d
        sz = int(min(ceil(2.0*n/minLeaf), 2**(maxDepth + 1)))   # pre-allocate storage for tree:
        self.L, self.R, self.F, self.T = np.zeros((sz,),dtype=int), np.zeros((sz,),dtype=int), np.zeros((sz,),dtype=int), np.zeros((sz,))
        sh = list(Y.shape)
        sh[0] = sz
        self.P = np.zeros(sh,dtype=Y.dtype) #np.zeros((sz,1))  # shape like Y 
        self.sz = 0              # start building at the root

        self.__train_recursive(X, Y, 0, minParent, maxDepth, minLeaf, nFeatures)

        self.L = self.L[0:self.sz]                              # store returned data into object
        self.R = self.R[0:self.sz]                              
        self.F = self.F[0:self.sz]
        self.T = self.T[0:self.sz]
        self.P = self.P[0:self.sz]

    def predict(self, X):
        return self.__predict_recursive(X, 0)
    def __train_recursive(self, X, Y, depth, minParent, maxDepth, minLeaf, nFeatures):
        n,d = mat(X).shape
        # check leaf conditions...
        if n < max(minParent,2*minLeaf) or depth >= maxDepth or np.var(Y-Y[0])==0: return self.__build_leaf(Y)
        best_val = np.inf
        best_feat = -1
        try_feat = np.random.permutation(d)
        # ...otherwise, search over (allowed) features
        for i_feat in try_feat[0:nFeatures]:
            dsorted = arr(np.sort(X[:,i_feat].T)).ravel()                # sort data...
            pi = np.argsort(X[:,i_feat].T)                               # ...get sorted indices...
            tsorted = Y[pi]                                              # ...and sort targets by feature ID
            can_split = np.append(arr(dsorted[:-1] != dsorted[1:]), 0)   # which indices are valid split points?
            # TODO: numeric comparison instead?
            can_split[np.arange(0,minLeaf-1)] = 0
            can_split[np.arange(n-minLeaf,n)] = 0   # TODO: check

            if not np.any(can_split):          # no way to split on this feature?
                continue

            # find min weighted variance among split points
            val,idx = self.data_impurity(tsorted, can_split)

            # save best feature and split point found so far
            if val < best_val:
                best_val, best_feat, best_thresh = val, i_feat, (dsorted[idx] + dsorted[idx + 1]) / 2.0

        # if no split possible, output leaf (prediction) node
        if best_feat == -1: return self.__build_leaf(Y)
        # split data on feature i_feat, value (tsorted[idx] + tsorted[idx + 1]) / 2
        self.F[self.sz] = best_feat
        self.T[self.sz] = best_thresh
        go_left = X[:,self.F[self.sz]] < self.T[self.sz]  # index data going left & right
        go_right= np.logical_not(go_left)
        my_idx = self.sz      # save current node index for left,right pointers
        self.sz += 1          # advance to next node to build subtree

        # recur left
        self.L[my_idx] = self.sz    
        self.__train_recursive(X[go_left,:], Y[go_left], depth+1, minParent, maxDepth, minLeaf, nFeatures)

        # recur right
        self.R[my_idx] = self.sz    
        self.__train_recursive(X[go_right,:], Y[go_right], depth+1, minParent, maxDepth, minLeaf, nFeatures)

        return
    def __predict_recursive(self, X, pos):
        """Recursive helper function for finding leaf nodes during prediction """
        m,n = X.shape
        sh = list(self.P.shape)
        sh[0] = m
        Yhat = np.zeros(sh,dtype=self.P.dtype)

        if self.F[pos] == -1:        # feature to compare = -1 => leaf node
            Yhat[:] = self.P[pos]    # predict stored value
        else:
            go_left = X[:,self.F[pos]] < self.T[pos]  # which data should follow left split?
            Yhat[go_left]  = self.__predict_recursive(X[go_left,:],  self.L[pos])
            go_right = np.logical_not(go_left)        # other data go right:
            Yhat[go_right] = self.__predict_recursive(X[go_right,:], self.R[pos])

        return Yhat

    def __build_leaf(self, Y):
        """Helper function for setting parameters at leaf nodes during train"""
        self.F[self.sz] = -1
        self.P[self.sz] = self.data_average(Y)      # TODO: convert to predict f'n call
        self.sz += 1


# In[4]:


class treeClassify(treeBase,classifier):
    def __init__(self, *args, **kwargs):
        """Constructor for decision tree regressor; all args passed to train"""
        self.classes = []
        treeBase.__init__(self,*args,**kwargs);
        #super(treeClassify,self).__init__(*args,**kwargs);

    def train(self, X, Y, *args,**kwargs):
        self.classes = list(np.unique(Y)) if len(self.classes) == 0 else self.classes
        treeBase.train(self,X,to1ofK(Y,self.classes).astype(float),*args,**kwargs);

    def predict(self,X):
        return classifier.predict(self,X)

    def predictSoft(self,X):
        return treeBase.predict(self,X);

    @staticmethod
    def entropy(tsorted, can_split):
        """Return the value and index of the minimum of the Shannon entropy impurity score"""
        n = tsorted.shape[0]
        eps = np.spacing(1)
        #y_left = np.cumsum(to1ofK(tsorted, self.classes), axis=0).astype(float)
        y_left = np.cumsum(tsorted, axis=0)
        y_right = y_left[-1,:] - y_left         # construct p(class) for each possible split
        wts_left = np.arange(1.0,n+1)     # by counting & then normalizing by left/right sizes
        y_left /= wts_left.reshape(-1,1)
        tmp = n - wts_left
        tmp[-1] = 1
        y_right /= tmp.reshape(-1,1)
        wts_left /= n
        h_root  = -np.dot(y_left[-1,:], np.log(y_left[-1,:] + eps).T)
        h_left  = -np.sum(y_left * np.log(y_left + eps), axis=1)
        h_right = -np.sum(y_right * np.log(y_right + eps), axis=1)
        IG = h_root - (wts_left * h_left + (1.0-wts_left) * h_right)
        val = np.max((IG + eps) * can_split)
        idx = np.argmax((IG + eps) * can_split)
        return (h_root-val,idx)

    @staticmethod
    def weighted_avg(Y, reg=0.5):
        p = np.sum(Y,axis=0) + reg
        return p / p.sum()

    data_impurity = entropy
    data_average  = weighted_avg


# In[5]:


data = pd.read_csv('classification/child-health.csv')
data.head()


# In[6]:


data.fetal_health.value_counts()


# In[18]:


label, features = data.iloc[:, -1].values, data.iloc[:, :-1].values

# Min Max Normaillization
min_ = features.min()
max_ = features.max()
features = (features - min_) / (max_ - min_)


# In[19]:


def bootstrapData(X, Y=None, n_boot=None):

    nx,dx = twod(X).shape
    if n_boot is None: n_boot = nx
    idx = np.floor(np.random.rand(n_boot) * nx).astype(int)
    if Y is None: return X[idx,:]
    Y = Y.flatten()
    return (X[idx,:],Y[idx])


# ### Random Forest
# We'll create a set of 20 random trees, each with bootstrapping, and combine them into a random forest.

# In[20]:


def most_frequent(List):
    return max(set(List), key = List.count)


# In[21]:


Ntrees= 20
Trees = [0]*Ntrees
Ytrain_trees = np.zeros((np.size(label),3))
pred_label = np.zeros((np.size(label),Ntrees))

# Evaluate for up to 20 learners.
for i in range(Ntrees):
    Xb,Yb = bootstrapData(features, label)
    Trees[i] = treeClassify(Xb, Yb, maxDepth= 15, minLeaf=12, nFeatures=21)
    Ytrain_trees += Trees[i].predictSoft(features)
    pred_label[:,i] = Trees[i].predict(features)

preds = [most_frequent(i.tolist()) for i in pred_label]
print(f"Accuracy: {accuracy_score(label, preds)}, Loss: {log_loss(label, Ytrain_trees)}")


# In[22]:


loss, auc = list(), list()
for i in range(2, 20):
    Ntrees= 20
    Trees = [0]*Ntrees
    Ytrain_trees = np.zeros((np.size(label),3))
    pred_label = np.zeros((np.size(label),Ntrees))
    print(f'max_depth is: {i} -->')
    for i in range(Ntrees):
        Xb,Yb = bootstrapData(features, label)
        Trees[i] = treeClassify(Xb, Yb, maxDepth= i, minLeaf=12, nFeatures=21)
        Ytrain_trees += Trees[i].predictSoft(features)
        pred_label[:,i] = Trees[i].predict(features)
    preds = [most_frequent(i.tolist()) for i in pred_label]
    loss.append(log_loss(label, Ytrain_trees))
    auc.append(accuracy_score(label, preds))
    print(f"Accuracy: {auc[-1]}, Loss: {loss[-1]}\n")


# In[23]:


plt.plot(np.arange(2, 20), auc) 
plt.title("Accuracy")
plt.show()


# In[24]:


plt.plot(np.arange(2, 20), loss) 
plt.title("Loss")
plt.show()


# In[ ]:




