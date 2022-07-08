import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
    
def train_valid_loaders(dataset, train_split=0.8, seed=42):
    """
    Divide a dataset into training and validation datasets; returns, for both of them, a np.array.

    Args:
        dataset (np.array): A dataset.
        train_split (float): A number between 0 and 1, corresponding to the 
            proportion of examples in the training dataset.
        seed (int): The used random seed.

    Returns:
        Tuple (train np.array, validation np.array).
    """
    number_of_data = len(dataset)
    indices = np.arange(number_of_data)
    np.random.seed(seed)
    np.random.shuffle(indices)    
    split = math.floor(train_split * number_of_data)
    train_idx, valid_idx = indices[:split], indices[split:]
    return dataset[train_idx], dataset[valid_idx]

def write(file_name, algo, dataset, seed, architecture, num_non_zero_param, max_repl_pr_epoch, C, tot):
    """
    Writes in a .txt file the hyperparameters and results of a training of the BGN algorithm 
        on a given dataset.

    Args:
        file_name (str): The name of the .txt file to write into.
        algo (str): Name of the used algorithm.
        dataset (str): Name of the dataset.
        seed (int): Used seed.
        architecture (str of the form "xx-xx-xx-..."): Architecture of the hidden layers.
            Ex. 100-150-50 means the networks has 3 hiddens layers, of respective sizes 100, 150 and 50.
        num_non_zero_param (int): Number of non-zero parameters in the network.
        max_repl_pr_epoch (int): Maximum number of hyperplanes to be removed and 
            replaced during a single iteration of the algorithm.
        C (float): Current regularization parameter for the Lasso regression.
        tot (list of floats): Train, dalid and test error.
    """
    file = open("results/"+str(file_name)+".txt","a")
    file.write(algo+'\t'+dataset+'\t'+str(seed)+'\t'+str(architecture)+
               '\t'+str(num_non_zero_param)+'\t'+str(max_repl_pr_epoch)+'\t'+str(C)+'\t'+str(tot[0])+"\t"+
               str(tot[1])+"\t"+str(tot[2])+"\n")
    file.close()
        
def find_threshold(data, dist) :
    """
    Completes the heuristic method of using Linear regression (and its variations; 
        Lasso, Ridge, etc.) for greedily parametrizing the neurons of a BNN. While 
        the regression is used to assign weights values, this function assignes bias values.
    
    There are n+1 possible separations the oriented hyperplane can make, given
        a dataset containing n examples; iteratively tries each of them, calculate
        the obtained MSE of the predictor; returns the one minimizing the obtained train MSE.

    Args:
        data (np.array): Label values of a training dataset.
        dist (np.array): Distance of the examples of a given dataset to a given hyperplane.
        
    Returns:
        Float.
    """
    inds = np.argsort(dist)
    dist, data = dist[inds], data[inds]
    data -= np.mean(data)
    n_data = len(data)
    mse = []
    
    #_1 corresponds to points on one side of the hyperplane, 
    #    _2 to points on the other side 
    
    mean_1 = [data[0]] 
    mean_2 = [np.mean(data[1:])]
    len_1 = [1]
    len_2 = [n_data-1]
    
    # The obtained MSE when using constant and
    #      independent predictors on both side of 
    #      the HP is calculated.
    
    for i in range(1,n_data-1) :
        mean_1.append((mean_1[i-1] * (i) + data[i]) / (i + 1))
        mean_2.append((mean_2[i-1] * (n_data - i) - data[i]) / (n_data - i - 1))
        len_1.append(i)
        len_2.append(n_data-i)
    for i in range(n_data-1) :
        mse.append(mean_1[i] ** 2 * len_1[i] + mean_2[i] ** 2 * len_2[i])
    ind = np.argmax(np.array(mse))
    return (dist[ind] + dist[ind+1]) / 2

def expand_dataset_dy(train, dy):
    """
    Expand the dataset according to the dimension of the label space (see the BGN
        paper, Improvement 3).
    
    Args:
        train (np.array): Train dataset
        dy (int): Dimension of the label space.
        
    Returns:
        np.array
    """
    train_long = np.hstack((train[:,:-dy].copy(), np.reshape(train[:,-dy], (-1,1))))
    for l in range(1,dy) :
        train_long = np.vstack((train_long, np.hstack((train[:,:-dy].copy(), np.reshape(train[:,-(dy-l)], (-1,1))))))
    return train_long

def optimal_cd(y_t, red_train, dy):
    """
    Computes the optimal values for c_t, d_t (Algorithm 1, line 7).
    
    Args:
        y_t (np.array): Train labels.
        red_train (np.array): Train feature's redescription.
        dy (int): Dimension of the label space.
        
    Returns:
        Tuple of np.array
    """
    inds_1 = np.nonzero(red_train[:,-1] + 1)
    inds_2 = np.nonzero(red_train[:,-1] - 1)
    c_t = (np.sum(y_t[inds_1[0],0]) / len(inds_1[0]) - np.sum(y_t[inds_2[0],0]) / len(inds_2[0])) / 2
    d_t = (np.sum(y_t[inds_1[0],0]) / len(inds_1[0]) + np.sum(y_t[inds_2[0],0]) / len(inds_2[0])) / 2
    for l in range(1,dy) :
        c_t = np.hstack((c_t, (np.sum(y_t[inds_1[0],l]) / len(inds_1[0]) - np.sum(y_t[inds_2[0],l]) / len(inds_2[0])) / 2))
        d_t = np.hstack((d_t, (np.sum(y_t[inds_1[0],l]) / len(inds_1[0]) + np.sum(y_t[inds_2[0],l]) / len(inds_2[0])) / 2))
    return c_t, d_t

def normalize_input_space(train, valid, test, dy):
    """
    Normalize the datasets according the to mean and stp of the train dataset.
    
    Args:
        train (np.array): Train dataset
        valid (np.array): Validation dataset
        test (np.array): Test dataset
        dy (int): Dimension of the label space.
        
    Returns:
        Tuple of np.array
    """
    mea = np.mean(train[:,0:-dy], axis = 0)
    std = np.std(train[:,0:-dy], axis = 0) + 1e-10
    train[:,0:-dy] = (train[:,0:-dy] - mea) / std
    valid[:,0:-dy] = (valid[:,0:-dy] - mea) / std
    test[:,0:-dy] = (test[:,0:-dy] - mea) / std
    return train, valid, test

def reset_to_last(hist, red, last, d_t):
    """
    When a remove-replace iteration doesn't lead to a lower training error, than
        the HP that has been removed is replaced, and the new one is removed.
    
    Args:
        hist (dictionnary): History of various values
        red (dictionnary): Redescription of the feature space
        last (np.array): Values concerning a HP that just has been removed
        d_t (int): Bias associated to the new HP
    """
    hist['non_zero_param'] = hist['non_zero_param'][:-1]
    hist['non_zero_param'].append(last['non_zero_param'])
    red['train_with_bias'][:,:-2], red['train_with_bias'][:,-2] = last['red_train_with_bias'][:,1:-1], last['red_train_with_bias'][:,0]
    red['train'][:,:-1], red['train'][:,-1] = last['red_train'][:,1:], last['red_train'][:,0]
    red['valid'][:,:-1], red['valid'][:,-1] = last['red_valid'][:,1:], last['red_valid'][:,0]
    red['test'][:,:-1], red['test'][:,-1] = last['red_test'][:,1:], last['red_test'][:,0]
    hist['param'][:-2,:], hist['param'][-2,:] = last['param'][1:-1,:], last['param'][0,:]
    hist['param'][-1] -= d_t

def fix_last(hist, red, last, param):
    """
    Fixes various values in the "last" dictionnary.
    
    Args:
        hist (dictionnary): History of various values
        red (dictionnary): Redescription of the feature space
        last (np.array): Values concerning a HP that just has been removed
        param (int): Number of non-zero parameters of the current HP
    """
    last['red_train_with_bias'] = red['train_with_bias'].copy()
    last['red_train'] = red['train'].copy()
    last['red_valid'] = red['valid'].copy()
    last['red_test'] = red['test'].copy()
    last['param'] = hist['param'].copy()
    hist['non_zero_param'].append(param.copy())
    
def obtain_perf(hist, red, dy, train, valid=None, test=None):
    """
    Returns the performances of the current network on train, valid and test datasets.
    
    Args:
        hist (dictionnary): History of various values
        red (dictionnary): Redescription of the feature space
        dy (int): Dimension of the label space.
        train (np.array): Train dataset
        valid (np.array): Validation dataset
        test (np.array): Test dataset
        
    Returns:
        Tuple of float
    """
    tr_err = np.mean(np.sum((train[:,-dy:]-np.matmul(np.hstack((red['train'], np.reshape(np.ones(len(red['train'])), (-1,1)))), hist['param'])) ** 2, axis=1))
    vd_err = np.mean(np.sum((valid[:,-dy:]-np.matmul(np.hstack((red['valid'], np.reshape(np.ones(len(red['valid'])), (-1,1)))), hist['param'])) ** 2, axis=1)) if valid is not None else 0
    te_err = np.mean(np.sum((test[:,-dy:]-np.matmul(np.hstack((red['test'], np.reshape(np.ones(len(red['test'])), (-1,1)))), hist['param'])) ** 2, axis=1)) if test is not None else 0
    return tr_err, vd_err, te_err
    
def print_perf(tr_err, vd_err, te_err, architecture, hist):
    """
    Display the current hidden architecture of the network and the performances
    
    Args:
        tr_err (float): Training error
        vd_err (float): Validation error
        te_err (float): Test error
        architecture (str): Architecture of the preceding hidden layers of the BNN.
        hist (dictionnary): History of various values
    """
    print('Hid. arch.: '+architecture+str(len(hist['param'])-1))
    print('Train loss: '+str(round(tr_err,4)))
    print('Valid loss: '+str(round(vd_err,4)))
    print('Test loss:  '+str(round(te_err,4)))
    print()

def update_y_t(hist, red, train, dy):
    """
    Updates the labels of the training dataset.
    
    Args:
        hist (dictionnary): History of various values
        red (dictionnary): Redescription of the feature space
        train (np.array): Original training dataset
        dy (int): Dimension of the label space.
     
    Returns:
        np.array
    """
    return np.reshape(train[:,-dy:] - np.matmul(red['train_with_bias'],hist['param']), -1, order='F')

def remove_hp(hist, red, train, valid, test, dy, last, y_t):
    """
    Updates the labels of the training dataset.
    
    Args:
        hist (dictionnary): History of various values
        red (dictionnary): Redescription of the feature space
        train (np.array): Train dataset
        valid (np.array): Validation dataset
        test (np.array): Test dataset
        dy (int): Dimension of the label space.
        last (np.array): Values concerning a HP that just has been removed
        y_t (np.array): Modified labels of the training dataset
    """
    tr_err, vd_err, te_err = obtain_perf(hist, red, dy, train, valid, test)
    last['tr_loss'] = tr_err.copy()
    red['train_with_bias'] = red['train_with_bias'][:,1:]
    red['train'] = red['train'][:,1:]
    red['valid'] = red['valid'][:,1:]
    red['test'] = red['test'][:,1:]
    last['non_zero_param'] = hist['non_zero_param'][0].copy()
    hist['non_zero_param'] = hist['non_zero_param'][1:]
    hist['param'] = hist['param'][1:]
    
def display_evolution(hist):
    """
    Shows the evolution of the train, valid and test loss as a function of the 
        number of neurons.
    
    Args:
        hist (dictionnary): History of various values
    """
    plt.xlabel('Number of neurons')
    plt.ylabel('Mean squared error')
    plt.title('Performances')
    plt.plot(np.array(range(1,len(hist['tr_loss'])+1)), np.array(hist['tr_loss']))
    plt.plot(np.array(range(1,len(hist['vd_loss'])+1)), np.array(hist['vd_loss']))
    plt.plot(np.array(range(1,len(hist['te_loss'])+1)), np.array(hist['te_loss']))
    plt.legend(['GBN 4.0 - Train',
                'GBN 4.0 - Valid',
                'GBN 4.0 - Test'])
    plt.xlim(0,len(hist['tr_loss'])+1)
    plt.ylim(0,max(max(hist['tr_loss'][2:]), max(hist['vd_loss'][2:]), max(hist['te_loss'][2:])))
    plt.show()
    
def load_dataset(dataset, sep, randseed):
    """
    Load a dataset.
    
    Args:
        dataset (str): Dataset name
        sep (float): Proportion of the training set to use as validation
        randseed (int): A random seed
        
    Return:
        Tuple (np.array, np.array, int)
    """
    np.random.seed(randseed)
    if dataset == 'bike_day' :
        data = pd.read_csv("datasets/bike.sharing.day.csv", sep=",", header=None)
        data = np.array(np.array(data)[1:,2:-2], dtype = float)
        dy = 1
    elif dataset == 'bike_hour' :
        data = pd.read_csv("datasets/bike.sharing.hour.csv", sep=",", header=None)
        data = np.delete(np.array(np.array(data)[1:,2:], dtype = float), [-2,-3], axis = 1)
        dy = 1
    elif dataset == 'carbon' :
        data = np.array(pd.read_csv("datasets/carbon_nanotubes.csv", sep=",", header=None))
        dy = 3
    elif dataset == 'diabete' :
        data = load_diabetes()
        X = data['data']
        y = data['target']
        data = np.hstack((X, np.reshape(y, (-1,1))))
        dy = 1
    elif dataset == 'housing' :
        data = np.array(pd.read_csv("datasets/housing.csv", sep=",", header=None))
        dy = 1
    elif dataset == 'hung_pox' :
        data = np.array(pd.read_csv("datasets/hungary.chickenpox.csv", sep=",", header=None))[1:,1:]        
        for i in range(len(data)) :
            data[i,-1] = data[i,-1][0]
        data = np.array(data, dtype = float)
        data = np.hstack((data[:,-3:], data[:,0:-3]))
        dy = 1
    elif dataset == 'ist_stock_usd' :
        data = np.array(pd.read_csv("datasets/istanbul.stock.usd.csv", sep=",", header=None))[:-2]
        data[:, [1, 0]] = data[:, [0, 1]]
        data = np.array(data, dtype = float)
        data[:,4:] *= 100
        dy = 1
    elif dataset == 'ist_stock_tl' :
        data = np.array(pd.read_csv("datasets/istanbul.stock.tl.csv", sep=",", header=None))[:-2]
        data[:, [1, 0]] = data[:, [0, 1]]
        data = np.array(data, dtype = float)
        data[:,4:] *= 100
        dy = 1
    elif dataset == 'parking' :
        data = np.array(pd.read_csv("datasets/parking.birmingham.csv", sep=",", header=None))[1:,:-1]    
        data[:, [1, 0]] = data[:, [0, 1]]    
        uni = np.unique(data[:,-3]).tolist()
        code = []
        cnt = 0
        for i in range(len(data)) :
            if data[i,-3] == uni[0] :
                code.append(cnt) 
            else :
                uni.pop(0)
                cnt += 1
                code.append(cnt) 
        data = np.hstack((np.reshape(np.array(code), (len(data),1)), data[:,1:]))
        data = np.hstack((data[:,:-3], data[:,-2:]))
        data = np.array(data, dtype = float)
        dy = 1
    elif dataset == 'power_plant' :
        data = np.array(pd.read_csv("datasets/power_plant.csv", sep=",", header=None))
        dy = 1
    elif dataset == 'solar_flare' :
        data = np.array(pd.read_csv("datasets/solar_flare.csv", sep=",", header=None, dtype=float))
        dy = 3
    elif dataset == 'portfolio' :
        data = np.array(pd.read_csv("datasets/stock_portfolio.csv", sep=",", header=None))
        dy = 6
    elif dataset == 'turbine' :
        data = np.array(pd.read_csv("datasets/turbine.csv", sep=",", header=None))
        dy = 2
    args = np.random.choice(len(data), int(len(data)*sep), replace=False)
    tr = data[args]
    te = data[~np.in1d(range(len(data)),args)]
    if dataset == 'housing' :
        tr[:,-1] /= 100000
        te[:,-1] /= 100000
    return tr, te, dy
