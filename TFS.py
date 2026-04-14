'''
author: liu qiong
'''

import numpy as np
import scipy.linalg
import torch as th
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import scipy


def load_data(url):
    data = pd.read_csv(url, header=None)
    X = data.values[0:, 0:-1]
    y = data.values[0:, -1]
    return X, y


def min_max_normal(data):
    maxmin = preprocessing.MinMaxScaler()
    all_data = maxmin.fit_transform(data)

    return all_data


class Model:
    
    def __init__(self, X, y, k, d, alpha, beta, gama, lam, mu: th.Tensor):
        '''
        X: np.array, shape = (n_samples, n_features)
        y: np.array, shape = (n_samples, )
        k: the number of rules with tsk fuzzy system
        d: the number of low-dimensional features
        alpha, beta: parameters of the model
        return:
        '''
        self.X = X
        self.y = y.reshape(-1, 1)
        self.k = k
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.mu = mu
        self.lam = lam
    
    
    def cal_fF(self, X_hat_d, P_d, p0_detached, F, k1):
        
        L = (X_hat_d @ P_d) * F @ k1 + F @ p0_detached - self.y
        f = th.sum(L**2) + th.sum(self.mu / F)    
        
        return f
    
    def cal_fP(self, X_hat_d, P, p0_detached, F_d, k1):
        
        L = (X_hat_d @ P) * F_d @ k1 + F_d @ p0_detached - self.y
        p = th.sum(L**2) + self.gama * th.linalg.norm(P, 'fro')**2
        
        return p
    
    def cal_fp0(self, X_hat_d, P_d, p0, F_d, k1):
        
        L = (X_hat_d @ P_d) * F_d @ k1 + F_d @ p0 - self.y
        p = th.sum(L**2)
        
        return p
    
        
    def cal_fX_hat(self, X_hat, P_d, p0_detached, F_d, k1, Q):
        
        L = (X_hat @ P_d) * F_d @ k1 + F_d @ p0_detached - self.y
        loss4 = th.sum(L**2) + self.alpha * th.linalg.norm(self.X@Q - X_hat, 'fro')**2 + self.beta * th.sum(th.linalg.norm(Q, dim=1)) + self.gama * th.linalg.norm(P_d, 'fro')**2 + th.sum(self.mu / F_d)
        
        return loss4
    
     
    
    def train(self):
        '''
        train the model with the data
        '''
        n, m = self.X.shape
        iter_num = 50
        # initialize
        P = th.rand(self.d, self.k, requires_grad=True)
        F = th.rand(n, self.k, requires_grad=True)
        Q = th.rand(m, self.d)+2
        Q, _ = th.qr(Q)
        Q = Q[:, :self.d]
        X_hat = (self.X @ Q).requires_grad_(True)
        p0 = th.rand(self.k, 1)
        theta = th.zeros(n)
        k1 = th.ones((self.k, 1))
        
        optimizer4 = th.optim.Adam([X_hat], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer1 = th.optim.Adam([F], lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        optimizer2 = th.optim.Adam([P], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        for _ in range(iter_num):
            
            flag = False
            with th.no_grad():
                
                para_tmp = th.eye(self.d)
                I = th.eye(self.d)
                Z = th.eye(self.X.size(1))
                for _ in range(2):
                    
                    tmp = self.alpha*self.X.T@self.X + self.beta*Z
                    val1, vec1 = scipy.linalg.eigh(tmp.numpy())
                    val2, vec2 = scipy.linalg.eigh(para_tmp.numpy())
                    val1 = th.from_numpy(val1).float()
                    val2 = th.from_numpy(val2).float()
                    vec1 = th.from_numpy(vec1).float()
                    vec2 = th.from_numpy(vec2).float()
                    C = self.alpha*self.X.T@X_hat
                    C_hat = vec1.T @ C @ vec2
                    Q_hat = C_hat / (val1.unsqueeze(1).repeat(1,vec2.size(1)) + val2.unsqueeze(0).repeat(vec1.size(1),1))
                    Q = vec1 @ Q_hat @ vec2.T
                    Z = th.diag(th.sqrt(th.sum(Q**2, dim=1))/2)
                    
                    para_tmp = para_tmp + 0.1*(th.diag(th.diag(Q.T @ Q - I)))
                    # 特征值分解可能存在数值不稳定的问题，导致出现极小的负数特征值
                    if (val1 < 0).any():
                        flag = True
                        break
                    elif (val2 < 0).any():
                        flag = True
                        break
            if flag:
                break        

            optimizer1.zero_grad()
            loss1 = self.cal_fF(X_hat.detach(), P.detach(), p0, F, k1, theta)
            loss1.backward()
            optimizer1.step()
            if (F<0).any():
                # 防止极小负数
                F[F<0] = 1e-6
            
            
            optimizer2.zero_grad()
            loss2 = self.cal_fP(X_hat.detach(), P, p0, F.detach(), k1, theta)
            loss2.backward()
            optimizer2.step()
            
            
            try:
                p0 = th.from_numpy(scipy.linalg.pinv(F.detach().numpy()) @ (self.y.numpy() - ((X_hat.detach().numpy() @ P.detach().numpy()) * F.detach().numpy()) @ k1.numpy())).float()
            except:
                break
            
            
            optimizer4.zero_grad()
            loss4 = self.cal_fX_hat(X_hat, P.detach(), p0, F.detach(), k1, Q, theta)
            loss4.backward()
            optimizer4.step()
            
            self.mu = 0.99*self.mu    

        return X_hat.detach(), Q, loss, F.detach()
                    

                   
if __name__ == '__main__':
    
    param_grid = {
        'alpha': [0.01, 0.1, 1],
        'beta': [0.01, 0.1, 1],  
        'gamma': [0.01, 0.1, 1],
    }
    import itertools
    params = list(itertools.product(*param_grid.values()))
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    XX, yy = load_data(urls)
    XX = min_max_normal(XX)
    count = 0

    param_optimal = {
        'alpha': [],
        'beta': [],  
        'gamma': [],
    }

    flag = 0
    d = int(0.3*XX.shape[1])
    ind = np.zeros((10, XX.shape[1]))
    for train_index, test_index in kf.split(XX, yy):

        X = XX[train_index]
        y = yy[train_index]
        n, m = X.shape
                    
        X = th.from_numpy(X).float()
        y = th.from_numpy(y).float()

        y_train, y_test = y.numpy(), yy[test_index]
        
        a_op = 0
        b_op = 0
        g_op = 0
        index_op = 0
        for alpha, beta, gamma in params:
            model = Model(X, y, k=len(th.unique(y)), d=d, alpha=alpha, beta=beta, gama=gamma, lam=0.1, mu=0.8)
        
            X_hat, P, loss, F = model.train()
            p = th.linalg.norm(P, dim=1)
            index = th.argsort(p, descending=True)
            
        



            
            
            
        