import numpy as np
import time
import scipy.linalg
import models



def one_multinom_sample(unnorm_probs):
    z = sum(unnorm_probs)
    sampled_unnorm_prob = np.random.uniform(0,z)
    cur_unnorm_prob = 0
    for i in range(len(unnorm_probs)):
        cur_unnorm_prob += unnorm_probs[i]
        if cur_unnorm_prob > sampled_unnorm_prob:
            return i
    print("MADE IT TO A WEIRD PLACE WTF")
    sys.exit()
    return len(unnorm_probs)-1
    
    
def build_rbf_kernel(D, sigma):
	gamma = .5/(sigma*sigma)
	return np.exp(-gamma*D)

def build_distance_sq_matrix(X, Z):
	return np.outer(np.sum(X**2, axis=1), np.ones(Z.shape[0])) -2*np.dot(X, Z.T) + np.outer(np.ones(X.shape[0]), np.sum(Z**2, axis=1))


def update_kernel_mtxs(train_kernel_mtx, test_to_train_kernel_mtx, X_train, X_test, new_point, sigma):

    new_point_val = X_test[np.array([new_point])]

    # add new point to train
    X_train = np.append(X_train, X_test[np.array([new_point])], axis=0)
    # remove new point from test
    X_test = np.delete(X_test, new_point, axis=0)

    # take row for new_point from test_to_train and put it in train_kernel
    new_point_to_train_kernel = np.asarray([test_to_train_kernel_mtx[new_point]])
    ## remove new_point distances from test_to_train
    test_to_train_kernel_mtx = np.delete(test_to_train_kernel_mtx, new_point, axis=0)
    new_col = np.append(new_point_to_train_kernel.T,[[1]], axis=0)
    with_new_row = np.concatenate((train_kernel_mtx, new_point_to_train_kernel))
    ## this guy is the new train_dist_mtx
    train_kernel_mtx = np.concatenate((with_new_row, new_col), axis=1)

    # compute distances between new point and test points
    test_to_new_point_dist_mtx = build_distance_sq_matrix(X_test, new_point_val)
    test_to_new_point_kernel_mtx = build_rbf_kernel(test_to_new_point_dist_mtx, sigma)
    # add new distances to test_to_train_dist_mtx
    test_to_train_kernel_mtx = np.concatenate((test_to_train_kernel_mtx, test_to_new_point_kernel_mtx), axis=1)

    return train_kernel_mtx, test_to_train_kernel_mtx, X_train, X_test


# to initialize the relevant matricies
def initialize_kernel_matrices(X_train, X_test, sigma):
    train_dist_mtx = build_distance_sq_matrix(X_train, X_train)
    train_kernel_mtx = build_rbf_kernel(train_dist_mtx, sigma)

    test_to_train_dist_mtx = build_distance_sq_matrix(X_test, X_train)
    test_to_train_kernel_mtx = build_rbf_kernel(test_to_train_dist_mtx, sigma)
    
    return train_kernel_mtx, test_to_train_kernel_mtx

def main():
    start_time = time.time()
    d = 1
    max_grid_size = 100001
    # std dev, for RBF kernel
    sigma = .001
    X = np.linspace(0,1,num=max_grid_size)
    
    X = np.array([np.array([xi]) for xi in X])
    
    new_point = np.random.randint(0,max_grid_size)
    X_train = X[np.array([new_point])]
    X_test = np.delete(X, new_point,axis=0)    
    
    k = 1000
    
    for i in range(k-1):
        start_iter_time = time.time()
        # update the kernel mtxs
        if i == 0:
            train_kernel_mtx, test_to_train_kernel_mtx = initialize_kernel_matrices(X_train, X_test, sigma)
        else:
            train_kernel_mtx, test_to_train_kernel_mtx, X_train, X_test = update_kernel_mtxs(train_kernel_mtx, 
                                                                           test_to_train_kernel_mtx, X_train, X_test, new_point, sigma)

        # compute unnorm probs        
        lower = True
        L = scipy.linalg.cholesky(train_kernel_mtx, lower)
        v = np.linalg.solve(L, test_to_train_kernel_mtx.T)
        unnorm_probs = 1-np.einsum('ij,ji->i', v.T,v)

        # get new point        
        new_point = one_multinom_sample(unnorm_probs)
        
        print("iteration: {}. this iter time: {}. total elapsed time: {}".format(i, 
                                                            round(time.time() - start_iter_time,2), round(time.time() - start_time,2)))

    
    
    X_train = np.append(X_train, X_test[np.array([new_point])], axis=0)
    return X_train

        
if __name__ == "__main__":
    main()
