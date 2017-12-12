import numpy as np
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


def update_kernel_mtxs(train_kernel_mtx, test_to_train_kernel_mtx, X_train, X_test, new_point_indx):
    # compute distance between new_point and existing training points
    # compute distance between new_point and test points
    new_point_val = X_test[np.array([new_point_indx])]
    print("about to print stuff:")
    print(build_distance_sq_matrix(new_point_val, X_train))


def main():
    d = 1
    max_grid_size = 11
    X = np.linspace(0,1,num=max_grid_size)
    
    X = np.array([np.array([xi]) for xi in X])
    
    new_point = np.random.randint(0,max_grid_size+1)
    X_train = X[np.array([new_point])]
    X_test = np.delete(X, new_point,axis=0)    
    

    

    train_kernel_mtx = np.array([1])
    L = None
    test_to_train_kernel_mtx = None
    #update_kernel_mtxs(train_kernel_mtx, test_to_train_kernel_mtx, X_train, X_test, new_point)
    

    k = 4
    for i in range(k-1):
        
        
        sigma_bw = .1
        sigma_noise = 0
        
        krr = models.kernel_ridge_regression(sigma_bw, sigma_noise)        
        krr.fit(X_train)
        
        update_kernel_mtxs(train_kernel_mtx, test_to_train_kernel_mtx, X_train, X_test, new_point)

        unnorm_probs = krr.predict(X_test)
        
        new_point = one_multinom_sample(unnorm_probs)
        new_X_train = np.append(X_train, X_test[np.array([new_point])], axis=0)
        new_X_test = np.delete(X_test, new_point, axis=0)
        
        
        X_train = new_X_train
        X_test = new_X_test
        print("iteration: {}".format(i))
        #print("X_train: {}".format(X_train))
        print("")


        
if __name__ == "__main__":
    #import cProfile
    #cProfile.run('main()', sort='cumtime')
    main()
