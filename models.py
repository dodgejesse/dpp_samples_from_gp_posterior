#this script was adapted from something written by Kevin Jamieson by Jesse Dodge

import numpy as np
import scipy.linalg

def build_rbf_kernel(D, sigma):
	gamma = .5/(sigma*sigma)
	return np.exp(-gamma*D)

def build_distance_sq_matrix(X, Z):
	return np.outer(np.sum(X**2, axis=1), np.ones(Z.shape[0])) -2*np.dot(X, Z.T) + np.outer(np.ones(X.shape[0]), np.sum(Z**2, axis=1))

class kernel_ridge_regression(object):
	def __init__(self, sigma_bw, sigma_noise=1e-4, sigma_f=1.):
		self.sigma_bw = sigma_bw
		self.sigma_f = sigma_f
		self.sigma_noise = sigma_noise
		self.L = None
		self.alpha = None
		self.X = None
		self.y_offset = 0.

	def fit(self, X):#, y):
		#self.y_offset = np.mean(y)
		self.X = X
		D_train = build_distance_sq_matrix(X, X)
                print("D_train: {}".format(D_train))
                print("D_train.shape: {}".format(D_train.shape))
		K_train = self.sigma_f*self.sigma_f*build_rbf_kernel(D_train, self.sigma_bw)

		lower = True
		K_22 = K_train + self.sigma_noise*self.sigma_noise*np.eye(X.shape[0])
                print(K_22[0])
                print(np.linalg.slogdet(K_22))
		self.L = scipy.linalg.cholesky(K_22, lower)


                # we're not computing the posterior mean, so we don't need this
		#alpha_tmp = np.linalg.solve(self.L, y-self.y_offset)
		#self.alpha = np.linalg.solve(self.L.T, alpha_tmp)

		# this code is doing the equivalent of
		# self.alpha = np.linalg.solve(K_22, y)

	def predict(self, X):
		D_12 = build_distance_sq_matrix(X, self.X)

		K_12 = self.sigma_f*self.sigma_f*build_rbf_kernel(D_12, self.sigma_bw)
                

		#D_11 = build_distance_sq_matrix(X, X)
		#K_11 = self.sigma_f*self.sigma_f*build_rbf_kernel(D_11, self.sigma_bw)
        

		v = np.linalg.solve(self.L, K_12.T)

		#mu_1 = np.dot(K_12, self.alpha)+self.y_offset

                
                Sigma_11 = 1-np.einsum('ij,ji->i', v.T,v)
                
		#Sigma_11 = 1 - np.dot(v.T, v)
		return Sigma_11 #mu_1, Sigma_11
