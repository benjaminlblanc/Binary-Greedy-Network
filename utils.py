from os.path import join
import itertools as it
from time import time
import math
from sklearn.datasets import load_diabetes, make_moons, fetch_openml
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from warnings import simplefilter
from sklearn.linear_model import *
from sklearn.model_selection import ParameterGrid
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)
simplefilter("ignore", category=ConvergenceWarning)


def create_layer(dataset,
                 dataset_name,
                 seed,
                 folds,
                 fold_num,
                 max_neur,
                 architecture,
                 res_param,
                 max_repl_pr_epoch,
                 max_coef_per_neur,
                 patience,
                 dy,
                 file_name,
                 weighting):
    """
    Given a dataset, iteratively builds a BNN's layer for classification or
        regression purposes using the BGN algorithm. Returns a dataset corresponding
        to the position of the examples in the hyperplane arrangement created
        by the hidden layer.

    Args:
        dataset (Tuple of n_examples x n_features np.arrays): Train, validation and test datasets.
        dataset_name (str): The dataset name.
        seed (int): A random seed to use.
        max_neur (int): Maximum number of neurons in the layer.
        architecture (str): Architecture of the preceding hidden layers of the BNN.
        res_param (int): Number of non-zero parameters in the preceding layers.
        max_repl_pr_epoch (int): Maximum number of hyperplanes to be removed and
            replaced during a single iteration of the algorithm.
        dy (int): Dimension of the label space.
        file_name (str): The name of the .txt file to write into.

    Returns:
        Tuple (train np.array,
               validation np.array,
               test np.array,
               validation loss float,
               number of non-zero params int,)
    """
    train, valid, test, f_n = dataset[0], dataset[1], dataset[2], dataset[3]
    # It's been found that normalized feature space yields better results
    normalize_input_space(train, valid, test, dy, weighting)

    # Expand the dataset according to the dimension of the label space (see the BGN paper, Improvement 3)
    exp_train = expand_dataset_dy(train.copy(), dy)
    X_t = exp_train[:, :-1].copy()
    y_t = exp_train[:, -1].copy()
    weighting = False
    if weighting:
        tr_wei = exp_train[:, -2].copy()
        vd_wei = valid[:, -2].copy()
        te_wei = test[:, -2].copy()
    else:
        tr_wei = np.ones((len(y_t)))
        vd_wei = np.ones((len(valid)))
        te_wei = np.ones((len(test)))
    tr_var = np.mean((train[:, -1] - np.mean(train[:, -1])) ** 2 * tr_wei)
    vd_var = np.mean((valid[:, -1] - np.mean(valid[:, -1])) ** 2 * vd_wei)
    te_var = np.mean((test[:, -1] - np.mean(test[:, -1])) ** 2 * te_wei)
    C_min, C_max = 0, 1e5  # Regularization parameter
    n_train = len(train)
    n_valid = len(valid)
    n_test = len(test)
    n_features = len(X_t[0])

    glm_freq = PoissonRegressor(alpha=1e-4, solver="newton-cholesky")
    glm_freq.fit(X_t, y_t, sample_weight=tr_wei)
    print(f"Train R2: {round(1 - np.mean((y_t - glm_freq.predict(X_t)) ** 2 * tr_wei) / tr_var, 4)}")
    print(f"Valid R2: {round(1 - np.mean((valid[:, -1] - glm_freq.predict(valid[:, :-1])) ** 2 * vd_wei) / vd_var, 4)}")
    print(f"Test  R2: {round(1 - np.mean((test[:, -1] - glm_freq.predict(test[:, :-1])) ** 2 * te_wei) / te_var, 4)}")

    hist = {'non_zero_param': np.zeros(max_neur),  # ... number of non-zero parameters
            'tr_loss': [],  # ... training loss
            'vd_loss': [],  # ... validation loss
            'te_loss': [],  # ... test loss
            'wb': np.zeros((n_features + 1, max_neur)),  # ... w_t and b_t
            'cd': np.zeros((max_neur + 1))}  # ... c_t and d_t

    red = {'train': np.ones((n_train, max_neur + 1)),  # ... training feature space
           'valid': np.ones((n_valid, max_neur + 1)),  # ... validation feature space
           'test': np.ones((n_test, max_neur + 1))}  # ... test feature space

    last = {'non_zero_param': np.zeros(max_neur),
            'train': np.ones((n_train, max_neur + 1)),  # ... training feature space
            'valid': np.ones((n_valid, max_neur + 1)),  # ... validation feature space
            'test': np.ones((n_test, max_neur + 1)),  # ... test feature space
            'wb': np.zeros((n_features + 1, max_neur)),
            'cd': np.zeros((max_neur + 1)),
            'tr_loss': math.inf}
    tot_cnt = 0  # Keeps track of how many remove-replace iterations have been done
    suc_cnt = 0  # Keeps track of how many remove-replace iterations have been done since last successful one
    curr_neur = 0  # Keeps track of the current neuron that is looking at
    curr_num_neur = 0  # Keeps track of how many hyperplanes have been placed
    best_vd_loss = math.inf
    w_t = np.array([0])
    if max_coef_per_neur == 0:
        max_coef_per_neur = len(X_t[0])
    while curr_num_neur < max_neur:
        # We use the Lasso regression to find the orientation of the current HP.
        ite = 0
        while C_max - C_min > 0.1 or np.sum(abs(w_t)) == 0 or np.sum(abs(w_t) > 0) > max_coef_per_neur:
            C = (C_max + C_min) / 2
            w_t = Lasso(alpha=C).fit(X=X_t, y=y_t / np.var(y_t) * 1000).coef_
            # If the regularization is too weak, it is upgraded a little.
            if np.sum(abs(w_t) > 0) > max_coef_per_neur:
                C_min = C
            # If the regularization is too strong, it is minimized a little.
            else:
                C_max = C
            ite += 1
            if ite == 50:
                C_min, C_max = 0, 1e10

        w_t[abs(w_t) > 0] = LinearRegression().fit(X=X_t[:, abs(w_t) > 0], y=y_t / np.var(y_t) * 1000,
                                                   sample_weight=tr_wei).coef_
        hist['wb'][:-1, curr_neur] = w_t

        # Find the optimal bias of the current HP.
        b_t = -find_threshold(y_t, np.sum(X_t * w_t, axis=1))
        hist['wb'][-1, curr_neur] = b_t

        # Keeps track of the number of non-zero parameters of the current HP.
        param = sum(abs(w_t) > 0)
        hist['non_zero_param'][curr_neur] = param
        # Every red_ numpy array corresponds to the output of the hidden layer we are currently building.
        #   It thus corresponds to the redescription of X.
        red['train'][:, curr_neur] = np.sign(np.sum(train[:, :-dy] * w_t, axis=1) + b_t)
        red['valid'][:, curr_neur] = np.sign(np.sum(valid[:, :-dy] * w_t, axis=1) + b_t)
        red['test'][:, curr_neur] = np.sign(np.sum(test[:, :-dy] * w_t, axis=1) + b_t)

        # Here, we find the optimal values for c_t and d_t.
        lin = LinearRegression().fit(X=red['train'][:, :curr_num_neur + 1],
                                     y=train[:, -1],
                                     sample_weight=tr_wei)
        c_t, d_t = lin.coef_, lin.intercept_
        hist['cd'][:curr_num_neur + 1] = c_t
        hist['cd'][curr_num_neur + 1] = d_t

        # The "if" is respected if a remove-replace iteration is not possible or relevant anymore.
        #   Then, the oldest HP won't be removed.
        if suc_cnt == curr_num_neur or tot_cnt > max_repl_pr_epoch:
            # If the last remove-replace iteration was successful, then nothing changes;
            #   if the error has not diminished, than the last HP is replaced.
            if last['tr_loss'] <= obtain_perf(hist, red, dy, curr_num_neur, tr_wei, vd_wei, te_wei, train, maxx=True)[
                0] + 1e-10:
                reset_to_last(hist, red, last)

            # We fix the parameters of the current neural network in the <last> dictionnary
            fix_last(hist, red, last)

            # We update y_t according to the new HP predictions
            y_t = train[:, -1] - LinearRegression().fit(X=red['train'][:, :curr_num_neur + 1],
                                                        y=train[:, -1],
                                                        sample_weight=tr_wei).predict(
                red['train'][:, :curr_num_neur + 1])
            # Performances are obtained, displayed, written and consigned
            tr_err, vd_err, te_err = obtain_perf(hist, red, dy, curr_num_neur, tr_wei, vd_wei, te_wei, train, valid,
                                                 test, maxx=True)[:3]
            print_perf(tr_err, vd_err, te_err, tr_var, vd_var, te_var, architecture, hist, curr_num_neur)
            R2 = 1 - te_err / np.var(test[:, -1])
            write(file_name, 'BGN', dataset_name, seed, folds, fold_num, architecture + str(curr_num_neur + 1),
                  np.sum(hist['wb'] != 0) + curr_num_neur + 1 + res_param,
                  len(np.unique(np.nonzero(hist['wb'])[1])) - 1, max_repl_pr_epoch, max_coef_per_neur,
                  C, patience, [tr_err, vd_err, te_err], R2)

            hist['tr_loss'].append(tr_err.copy())
            hist['vd_loss'].append(vd_err.copy())
            hist['te_loss'].append(te_err.copy())
            last['tr_loss'] = tr_err.copy()

            # A neuron is added
            tot_cnt = 0
            suc_cnt = 0
            curr_num_neur += 1
            curr_neur = curr_num_neur
            last['tr_loss'] = math.inf
            C_min /= 10
            C_max *= 10


        # The "if" would have been respected if a remove-replace iteration was not possible anymore.
        #   Therefore, entering the "else" means the oldest HP will be removed.
        else:
            # If the last remove-replace iteration was successful, than nothing changes;
            #   if the error has not diminished, than the last HP is replaced.
            curr_tr_loss = obtain_perf(hist, red, dy, curr_num_neur, tr_wei, vd_wei, te_wei, train)[0]
            if last['tr_loss'] <= curr_tr_loss + 1e-10:
                reset_to_last(hist, red, last)
                suc_cnt += 1
            else:
                # If the last remove-replace iteration was successul, than the counter is reset.
                last['tr_loss'] = curr_tr_loss.copy()
                suc_cnt = 0
            # We fix the parameters of the current neural network in the "last" dictionnary
            fix_last(hist, red, last)

            # An iteration of the remove-replace technique begins by removing the oldest HP
            hist['cd'] = np.zeros((max_neur + 1))

            # We update y_t once again, since a HP has been removed
            curr_neur = 0 if curr_neur == curr_num_neur else curr_neur + 1
            red['train'][:, curr_neur] = 0
            # y_t = update_y_t(hist, red, train, dy)
            y_t = train[:, -1] - LinearRegression().fit(X=red['train'][:, :curr_num_neur + 1],
                                                        y=train[:, -1],
                                                        sample_weight=tr_wei).predict(
                red['train'][:, :curr_num_neur + 1])
            tot_cnt += 1
            C_min /= 5
            C_max *= 5
        if curr_num_neur > 0:
            if best_vd_loss > hist['vd_loss'][-1]:
                best_tr_loss = hist['tr_loss'][-1].copy()
                best_vd_loss = hist['vd_loss'][-1].copy()
                best_te_loss = hist['te_loss'][-1].copy()
                best_red = red.copy()
                best_param = np.count_nonzero(hist['wb'].copy()) + res_param
                best_num_neur = curr_num_neur

        if curr_num_neur > patience:
            if min(hist['vd_loss'][-patience:]) > min(hist['vd_loss']):
                curr_num_neur = max_neur

    # np.savetxt(f'Mathieu_1/models/w_{dataset_name}.txt', np.transpose(best_wd[:,:-1] / std), delimiter=',')
    # np.savetxt(f'Mathieu_1/models/b_{dataset_name}.txt', np.squeeze(best_wd[:,-1] - np.sum(hist['wb'][:, :-1] / std * mean, axis=-1)), delimiter=',')
    # np.savetxt(f'Mathieu_1/models/c_{dataset_name}.txt', best_cb[:-1], delimiter=',')
    # np.savetxt(f'Mathieu_1/models/d_{dataset_name}.txt', best_cb[-1], delimiter=',')
    # pred = SimpleBANN(np.transpose(best_wd[:,:-1]), np.squeeze(best_wd[:,-1]), best_cb[:-1], best_cb[-1])

    # The evolution of the performances in training, validation and test are displayed

    # display_evolution(hist)
    # tr_err, vd_err, te_err = obtain_perf(hist, red, dy, train, valid, test, task, maxx=True)[:3]
    # print_perf(tr_err, vd_err, te_err, architecture, hist)

    # tr_preds = np.matmul(np.hstack((best_red['train'], np.reshape(np.ones(len(best_red['train'])), (-1, 1)))), best_cb)
    # vd_preds = np.matmul(np.hstack((best_red['valid'], np.reshape(np.ones(len(best_red['valid'])), (-1, 1)))), best_cb)
    # te_preds = np.matmul(np.hstack((best_red['test'], np.reshape(np.ones(len(best_red['test'])), (-1, 1)))), best_cb)

    # compute_importance(X_t, hist, f_n, mean, std, task)
    # compute_predictor(hist, mean, std, norm=False)
    # compute_predictor(hist, mean, std, norm=True)
    return 1 - best_tr_loss / tr_var, 1 - best_vd_loss / vd_var, 1 - best_te_loss / te_var, best_num_neur

class SimpleBANN():
    def __init__(self, w, b, c, d):
        assert len(w.shape) == 2, f'Expected w to be of shape (d,h), got {w.shape} (where h = number of hidden neurons).'
        assert len(b.shape) == 1, f'Expected b to be of shape (h), got {b.shape} (where h = number of hidden neurons).'
        assert len(c.shape) == 2, f'Expected c to be of shape (h,1), got {c.shape} (where h = number of hidden neurons).'
        assert len(d) == 1, f'Expected d to be of shape (1), got {d.shape}.'
        self.w, self.b, self.c, self.d = w, b, c, d

    def predict(self, x):
        assert len(self.w[0]) != len(x), f'Wrong feature dimension; expected {len(self.w)}, got {len(x[0])}.'
        hidden = np.sign(np.matmul(x, self.w) + self.b)
        return np.matmul(hidden, self.c) + self.d
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


def write(file_name, algo, dataset, seed, fold, fold_number, architecture, num_non_zero_param, num_inputs,
          max_repl_pr_epoch, max_coef, C, patience, tot, R2):
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
    file = open("results/" + str(file_name) + ".txt", "a")
    file.write(algo + '\t' + dataset + '\t' + str(seed) + '\t' + str(fold) + '\t' + str(fold_number) + '\t' + str(architecture) +
               '\t' + str(num_non_zero_param) + '\t' + str(num_inputs) + '\t' + str(max_repl_pr_epoch) + '\t' +
               str(max_coef) + '\t' + str(C) + '\t' + str(patience) + '\t' + str(tot[0]) + "\t" +
               str(tot[1]) + "\t" + str(tot[2]) + "\t" + str(R2) + "\n")
    file.close()


def find_threshold(data, dist, step=1):
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
    nnzero = np.nonzero(data)[0]
    dist = dist[nnzero]
    data = data[nnzero]

    inds = np.argsort(dist)
    dist, data = dist[inds], data[inds]
    data -= np.mean(data)
    n_data = len(data)
    mse = []

    mean_1 = [data[0]]
    mean_2 = [np.mean(data[1:])]
    len_1 = [1]
    len_2 = [n_data - 1]

    # The obtained MSE when using constant and
    #      independent predictors on both side of
    #      the HP is calculated.

    for i in range(1, n_data - 1, step):
        mean_1.append((mean_1[int(i / step) - 1] * (i) + sum(data[i:i + step])) / (i + step))
        mean_2.append((mean_2[int(i / step) - 1] * (n_data - i) - sum(data[i:i + step])) / (n_data - i - step))
        len_1.append(i + 1)
        len_2.append(n_data - i - 1)
    for i in range(0, n_data - 1, step):
        mse.append(
            mean_1[int(i / step)] ** 2 * len_1[int(i / step)] + mean_2[int(i / step)] ** 2 * len_2[int(i / step)])
    ind = int(np.argmax(np.array(mse)) * step)
    return (dist[ind] + dist[ind + 1]) / 2

def compute_predictor(hist, mean, std, norm):
    w = hist['wb'][:, :-1].copy()
    b = hist['wb'][:, -1].copy()
    if not norm:
        w /= std
        b -= np.sum(hist['wb'][:, :-1] / std * mean, axis=-1)
    b *= -1
    c = hist['param'][:-1].copy()
    d = hist['param'][-1][0].copy()
    d -= np.sum(c)
    c *= 2
    order = np.argsort(np.reshape(c, (-1)))[::-1]
    w = w[order]
    b = b[order]
    c = c[order]
    string = f"{round(d, 3)}~&+ \\\\"
    for i in range(len(w)):
        if c[i][0] < 0 and i > 0:
            #string = string[:-3]
            sub_string = f" - {abs(round(c[i][0], 3))} ~&\cdot "
        else:
            sub_string = f"{round(c[i][0],3)} ~&\cdot "
        sub_string += "\mathbbm{1}_{\{"
        cnt_j = 0
        switch = False
        if sum(abs(w[i]) > 0) > 1:
            magn = max(abs(w[i, abs(w[i]) > 0]))
            w[i, abs(w[i]) > 0] /= magn
            b[i] /= magn
            for j in range(len(w[0])):
                if w[i,j] < 0 :
                    if cnt_j > 0:
                        sub_string = sub_string[:-3]
                    sub_string += " - "
                if w[i,j] != 0:
                    cnt_j += 1
                    if w[i,j] == -1 or w[i,j] == 1:
                        sub_string += "x_{"+f"{j + 1}"+"} + "
                    else:
                        sub_string += f"{abs(round(w[i, j], 3))}"+" \cdot x_{"+f"{j+1}"+"} + "
        else:
            for j in range(len(w[0])):
                if w[i, j] < 0:
                    switch = True
                if w[i,j] != 0:
                    cnt_j += 1
                    b[i] /= w[i,j]
                    w[i, j] = 1
                    sub_string += "x_{"+f"{j+1}"+"} + "
        sub_string = sub_string[:-3]
        if switch:
            sub_string += f" < {round(b[i],3)}"
        else:
            sub_string += f" > {round(b[i],3)}"
        sub_string += "\}} &+ \\\\"
        string += sub_string
    print(string[:-6])

def compute_importance(X_t, hist, f_n, mean, std, task):
    t = time()
    if task == 'regression':
        imp = np.zeros((len(X_t[0]), len(X_t[0]), len(hist['wb'])))
        for i in range(len(imp)):  # Num. de la feature à considérer
            for j in range(len(hist['wb'])):  # Num. de l'hyperplan
                if hist['wb'][j, i] != 0:
                    imp[i] += get_comb_reg(i, j, hist, X_t)
        c_tot = sum_pond_reg(imp)
    elif task == 'classification':
        imp = np.zeros((len(X_t[0]), len(X_t[0])))
        for i in range(len(imp)):  # Num. de la feature à considérer
            if np.sum(np.abs(hist['wb'][:, i])) != 0:
                imp[i] += get_comb_classif(i, hist, X_t)
        c_tot = sum_pond_classif(imp)
    print(f"\nImportance - Took {round(time() - t, 2)} seconds to compute.\n")
    #print(c_tot / np.sum(c_tot, axis=0))
    print()
    if task == 'regression':
        fea_imp = np.sum(c_tot, axis=1) / np.sum(c_tot)
    elif task == 'classification':
        fea_imp = c_tot / np.sum(c_tot)
    #print(imp)
    #print(c_tot)
    for fea in range(len(fea_imp)):
        if np.sum(hist['wb'][:, fea]) != 0:
            print(f"Feature #{fea + 1}: {round(fea_imp[fea] * 100, 2)}% ({f_n[fea]})")
    print()
    if task == 'regression':
        hyp_imp = np.sum(c_tot, axis=0) / np.sum(c_tot)
        cnt = 1
        for hyp in np.argsort(np.reshape(hist['param'][:-1], (-1)))[::-1]:
            print(f"Hyperplane #{cnt}: {round(hyp_imp[hyp] * 100, 2)}%")
            cnt += 1
        print()


def get_comb_reg(i, j, hist, X_t):
    inds = np.nonzero(hist['wb'][j, :-1])[0]
    c = np.zeros((len(hist['wb'][0]) - 1, len(hist['wb'])))
    for k in range(len(inds)):  # Nb. de variables (autres que celle considérée) considérés
        print(
            f"Hyp. {j + 1}/{len(hist['wb'])} - Feature {i + 1}/{len(X_t[0])} - Nb. other features {k + 1}/{len(inds)}")
        combs = comb_adv(len(inds) - 1, k)
        for comb in combs:
            pos_i = sum(inds < i)
            if len(combs[0]) == 0:
                inds_mod_with = inds[True]
                inds_inv_with = inds[False]
                inds_mod_without = inds[False]
                inds_inv_without = inds[True]
            else:
                inds_mod_with = inds[np.insert(comb, pos_i, True).astype(bool)]
                inds_inv_with = inds[np.insert(comb * -1, pos_i, False).astype(bool)]
                inds_mod_without = inds[np.insert(comb, pos_i, False).astype(bool)]
                inds_inv_without = inds[np.insert(comb * -1, pos_i, True).astype(bool)]
            if len(inds_inv_with) != 0:
                exs_with = np.reshape(np.sum(X_t[:, inds_inv_with] * hist['wb'][j][inds_inv_with], axis=1), (-1))
            else:
                exs_with = np.zeros(len(X_t))
            if len(inds_inv_without) != 0:
                exs_without = np.reshape(np.sum(X_t[:, inds_inv_without] * hist['wb'][j][inds_inv_without], axis=1),
                                         (-1))
            else:
                exs_without = np.zeros(len(X_t))
            for ex in X_t:
                ex_with = exs_with.copy()
                ex_without = exs_without.copy()
                if len(inds_mod_with) != 0:
                    ex_with += np.sum(ex[inds_mod_with] * hist['wb'][j][inds_mod_with])
                if len(inds_mod_without) != 0:
                    ex_without += np.sum(ex[inds_mod_without] * hist['wb'][j][inds_mod_without])
                ex_with = np.mean(np.sign(ex_with + hist['wb'][j][-1])) * hist['param'][j]
                ex_without = np.mean(np.sign(ex_without + hist['wb'][j][-1])) * hist['param'][j]
                c[k, j] += np.abs(ex_with - ex_without)
    return c

def get_comb_classif(i, hist, X_t):
    inds = np.nonzero(np.sum(hist['wb'][:, :-1], axis=0))[0]
    c = np.zeros((len(hist['wb'][0]) - 1))
    for k in range(len(inds)):  # Nb. de variables (autres que celle considérée) considérés
        combs = comb_adv(len(inds) - 1, k)
        for comb in combs:
            cnt = -1
            for ex in X_t:
                cnt += 1
                ex_with_cum = np.zeros(len(X_t))
                ex_without_cum = np.zeros(len(X_t))
                pos_i = sum(inds < i)
                for j in range(len(hist['wb'])):
                    if len(combs[0]) == 0:
                        inds_mod_with = inds[True]
                        inds_inv_with = inds[False]
                        inds_mod_without = inds[False]
                        inds_inv_without = inds[True]
                    else:
                        inds_mod_with = inds[np.insert(comb, pos_i, True).astype(bool)]
                        inds_inv_with = inds[np.insert(comb * -1, pos_i, False).astype(bool)]
                        inds_mod_without = inds[np.insert(comb, pos_i, False).astype(bool)]
                        inds_inv_without = inds[np.insert(comb * -1, pos_i, True).astype(bool)]
                    if len(inds_inv_with) != 0:
                        exs_with = np.reshape(np.sum(X_t[:, inds_inv_with] * hist['wb'][j][inds_inv_with], axis=1), (-1))
                    else:
                        exs_with = np.zeros(len(X_t))
                    if len(inds_inv_without) != 0:
                        exs_without = np.reshape(np.sum(X_t[:, inds_inv_without] * hist['wb'][j][inds_inv_without], axis=1),
                                                 (-1))
                    else:
                        exs_without = np.zeros(len(X_t))
                    ex_with = exs_with.copy()
                    ex_without = exs_without.copy()
                    if len(inds_mod_with) != 0:
                        ex_with += np.sum(ex[inds_mod_with] * hist['wb'][j][inds_mod_with])
                    if len(inds_mod_without) != 0:
                        ex_without += np.sum(ex[inds_mod_without] * hist['wb'][j][inds_mod_without])
                    ex_with = np.sign(ex_with + hist['wb'][j][-1]) * hist['param'][j]
                    ex_without = np.sign(ex_without + hist['wb'][j][-1]) * hist['param'][j]
                    ex_with_cum += ex_with
                    ex_without_cum += ex_without
                c[k] += np.mean(np.abs(np.sign(ex_with_cum+hist['param'][-1])-np.sign(ex_without_cum+hist['param'][-1])))
    return c

def sum_pond_classif(c):
    c_tot = np.zeros((len(c[0])), dtype=float)
    for ite in range(len(c[0])):
        if np.sum(c[:, ite]) != 0:
            c_tot = c_tot + c[:, ite] * np.math.factorial(ite) / np.prod(list(range(len(c[0])-ite, len(c[0])+1)))
    return c_tot

def sum_pond_reg(c):
    c_tot = np.zeros((len(c[0]), len(c[0, 0])), dtype=float)
    for ite in range(len(c[0])):
        if np.sum(c[:, ite]) != 0:
            c_tot = c_tot + c[:, ite] * np.math.factorial(ite) / np.prod(list(range(len(c[0])-ite, len(c[0])+1)))
    return c_tot

def bino(n, k):
    base = np.math.factorial(n) / np.math.factorial(k) / np.math.factorial(n - k)
    other = bino(n, k - 1) if k > 0 else 0
    return int(base + other)


def comb_adv(n, k):
    array = np.array([0, 1])
    combinations = it.product(array, repeat=n)
    comb = []
    for combination in combinations:
        comb.append(combination)
    comb = np.array(comb)
    comb = comb[(comb == 0).sum(axis=1).argsort()]
    if n == k:
        return np.ones((1, n), dtype=bool)
    elif n == 0:
        return np.array([], dtype=bool)
    else:
        return comb[bino(n, n - k - 1):bino(n, n - k)]


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
    train_long = np.hstack((train[:, :-dy].copy(), np.reshape(train[:, -dy], (-1, 1))))
    for l in range(1, dy):
        train_long = np.vstack(
            (train_long, np.hstack((train[:, :-dy].copy(), np.reshape(train[:, -(dy - l)], (-1, 1))))))
    return train_long


def expand_dataset_cnn(X_t, y_t, filter_dims, feature_dims, stride=2):  # 0 padding?
    """
    Expand the dataset according to the calculation of a CNN layer.

    Args:
        train (np.array): Train dataset
        filter_dims (list): List of the dimensions of the filter
        feature_dims (list): list of the dimensions of the filter

    Returns:
        np.array
    """
    X_t = X_t.astype(int)
    y_t = y_t.astype(int)
    for it1 in range(0, feature_dims[1] - filter_dims[1] + 1, stride):
        d1 = int(it1 / stride) + 1
    for it2 in range(0, feature_dims[0] - filter_dims[0] + 1, stride):
        d2 = int(it2 / stride) + 1
    train_long = np.zeros((d1 * d2 * len(X_t),
                           filter_dims[2] * filter_dims[1] * filter_dims[0] + 1), dtype=int)
    cnt = 0
    for i in range(0, feature_dims[1] - filter_dims[1] + 1, stride):
        print(f'{i + 1} out of {feature_dims[0] - filter_dims[0] + 1}')
        for j in range(0, feature_dims[0] - filter_dims[0] + 1, stride):
            inds = []
            for k in range(filter_dims[2]):
                for l in range(filter_dims[1]):
                    for m in range(filter_dims[0]):
                        inds.append(m + j + (l + i) * feature_dims[0] + k * feature_dims[1] * feature_dims[0])
            cnt += 1
            train_long[
            (int(j / stride) + int(i / stride) * d1) * len(X_t):(int(j / stride) + 1 + int(i / stride) * d1) * len(
                X_t)] = np.hstack((X_t[:, inds], np.reshape(y_t, (-1, 1))))
    return train_long


def weights_cnn_to_weights_fc(weights_cnn, filter_dims, feature_dims, stride=2):
    """
    Expand the dataset according to the calculation of a CNN layer.

    Args:
        train (np.array): Train dataset
        filter_dims (list): List of the dimensions of the filter
        feature_dims (list): list of the dimensions of the filter

    Returns:
        np.array
    """
    for it1 in range(0, feature_dims[1] - filter_dims[1] + 1, stride):
        d1 = int(it1 / stride) + 1
    for it2 in range(0, feature_dims[0] - filter_dims[0] + 1, stride):
        d2 = int(it2 / stride) + 1
    weights_fc = np.zeros((d1 * d2, feature_dims[2] * feature_dims[1] * feature_dims[0]))
    for i in range(0, feature_dims[1] - filter_dims[1] + 1, stride):
        for j in range(0, feature_dims[0] - filter_dims[0] + 1, stride):
            for k in range(filter_dims[2]):
                for l in range(filter_dims[1]):
                    for m in range(filter_dims[0]):
                        weights_fc[int(j / stride) + int(i / stride) * d1, m + j + (l + i) * feature_dims[0] + k *
                                   feature_dims[1] * feature_dims[0]] = weights_cnn[
                            m + l * filter_dims[0] + k * filter_dims[1] * filter_dims[0]]
    return weights_fc

def adjust_weights(train, valid, test, red, c_0, c_1, print_perf=False):
        weights = np.ones(len(train))
        summ_1 = 0
        summ_2 = 0
        summ_3 = 0
        uniques = np.unique(red['train'], axis=0)
        if len(uniques) > 1:
            for uni in uniques:
                inds_tr = np.squeeze(np.all(red['train'] == uni, axis=1))
                inds_vd = np.squeeze(np.all(red['valid'] == uni, axis=1))
                inds_te = np.squeeze(np.all(red['test'] == uni, axis=1))
                mean = np.mean(train[inds_tr, -1])
                prop = ((np.abs(mean) + 1) / 2)
                weights[inds_tr] = prop + (1 - 2 * prop) * (1 + np.sign(train[inds_tr, -1]) * np.sign(mean)) / 2
                summ_1 += abs(sum((train[inds_tr, -1] + 1) / 2 / c_1 - 1))
                summ_2 += abs(sum((valid[inds_vd, -1] + 1) / 2 / c_1 - 1))
                summ_3 += abs(sum((test[inds_te, -1] + 1) / 2 / c_1 - 1))

        if print_perf:
            print(f"Perf. : {(sum((train[:, -1] + 1) / 2 * (1 / c_1 - 2) + 1) - summ_1) / sum((train[:, -1] + 1) / 2 * (1 / c_1 - 2) + 1) / 2}")
            print(f"Perf. : {(sum((valid[:, -1] + 1) / 2 * (1 / c_1 - 2) + 1) - summ_2) / sum((valid[:, -1] + 1) / 2 * (1 / c_1 - 2) + 1) / 2}")
            print(f"Perf. : {(sum((test[:, -1] + 1) / 2 * (1 / c_1 - 2) + 1) - summ_3) / sum((test[:, -1] + 1) / 2 * (1 / c_1 - 2) + 1) / 2}")
            print("-----")

        for i in range(len(weights)):
            if train[i,-1] == -1 :
                weights[i] /= c_0
            else:
                weights[i] /= c_1
        if sum(weights) > 0 :
            weights /= sum(weights)
            trigger = False
        else:
            trigger = True
        return weights, trigger

def find_w_t(X_t, y_t, weights, C_0, C_1, max_coef_per_neur):
    if max_coef_per_neur == 0:
        reg = LinearRegression()
    else:
        reg = Lasso(alpha=10 ** ((C_0 + C_1) / 2))
    w_t = reg.fit(X=X_t, y=y_t, sample_weight=weights).coef_

    if np.sum(abs(w_t) > 0) > max_coef_per_neur or np.sum(abs(w_t)) < 1e-5:
        # We use the Lasso regression to find the orientation of the current HP.
        reg = Lasso(alpha=10 ** C_0)
        w_t_0 = reg.fit(X=X_t, y=y_t, sample_weight=weights).coef_
        reg = Lasso(alpha=10 ** C_1)
        w_t_1 = reg.fit(X=X_t, y=y_t, sample_weight=weights).coef_
        while np.sum(abs(w_t_1)) > 1e-100:
            C_1 += 1
            reg = Lasso(alpha=10 ** C_1)
            w_t_1 = reg.fit(X=X_t, y=y_t, sample_weight=weights).coef_
        while np.sum(abs(w_t_0) > 0) <= max_coef_per_neur:
            C_0 -= 1
            reg = Lasso(alpha=10 ** C_0)
            w_t_0 = reg.fit(X=X_t, y=y_t, sample_weight=weights).coef_

        reg = Lasso(alpha=10 ** ((C_0 + C_1) / 2))
        w_t = reg.fit(X=X_t, y=y_t, sample_weight=weights).coef_
        n_it = 0
        n_re = 0
        while np.sum(abs(w_t) > 0) > max_coef_per_neur or np.sum(abs(w_t)) < 1e-5:
            n_it += 1
            if np.sum(abs(w_t)) < 1e-100:
                C_1 = (C_0 + C_1) / 2
            elif np.sum(abs(w_t) > 0) > max_coef_per_neur:
                C_0 = (C_0 + C_1) / 2
            if n_it >= 25:
                C_0, C_1 = -10, 10
                n_it = 0
                n_re += 1
                if n_re > 1 :
                    reg = LinearRegression()
                    w_t = reg.fit(X=X_t, y=y_t, sample_weight=weights).coef_
                    inds = np.argsort(np.abs(w_t))
                    w_t[inds[:-max_coef_per_neur]] *= 0
                    n_re = 0
                    break
            reg = Lasso(alpha=10 ** ((C_0 + C_1) / 2))
            w_t = reg.fit(X=X_t, y=y_t, sample_weight=weights).coef_
        w = LinearRegression().fit(X=X_t[:, abs(w_t) > 0], y=y_t, sample_weight=weights).coef_
        w_t[abs(w_t) > 0] = w
    return C_0, C_1, w_t

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
    inds_1 = np.nonzero(red_train[:, -1] + 1)
    inds_2 = np.nonzero(red_train[:, -1] - 1)
    c_t = (np.sum(y_t[inds_1[0], 0]) / len(inds_1[0]) - np.sum(y_t[inds_2[0], 0]) / len(inds_2[0])) / 2
    d_t = (np.sum(y_t[inds_1[0], 0]) / len(inds_1[0]) + np.sum(y_t[inds_2[0], 0]) / len(inds_2[0])) / 2
    for l in range(1, dy):
        c_t = np.hstack(
            (c_t, (np.sum(y_t[inds_1[0], l]) / len(inds_1[0]) - np.sum(y_t[inds_2[0], l]) / len(inds_2[0])) / 2))
        d_t = np.hstack(
            (d_t, (np.sum(y_t[inds_1[0], l]) / len(inds_1[0]) + np.sum(y_t[inds_2[0], l]) / len(inds_2[0])) / 2))
    return c_t, d_t


def normalize_input_space(train, valid, test, dy, weighting):
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
    max_val = 2 if weighting else 1
    mea = np.mean(train[:, :-max_val], axis=0)
    std = np.std(train[:, :-max_val], axis=0) + 1e-10
    train[:, :-max_val] = (train[:, :-max_val] - mea) / std
    valid[:, :-max_val] = (valid[:, :-max_val] - mea) / std
    test[:, :-max_val] = (test[:, :-max_val] - mea) / std
    return mea, std

def reset_to_best(hist, red, last, param):
    """
    Fixes various values in the "last" dictionnary.

    Args:
        hist (dictionnary): History of various values
        red (dictionnary): Redescription of the feature space
        last (np.array): Values concerning a HP that just has been removed
        param (int): Number of non-zero parameters of the current HP
    """
    red['train_with_bias'] = last['red_train_with_bias'].copy()
    red['train'] = last['red_train'].copy()
    red['valid'] = last['red_valid'].copy()
    red['test'] = last['red_test'].copy()
    hist['param'] = last['param'].copy()
    hist['wb'] = last['wb'].copy()
    hist['cd'] = last['cd'].copy()
    #last['non_zero_param'].append(param.copy())

def reset_to_last(hist, red, last):
    """
    When a remove-replace iteration doesn't lead to a lower training error, than
        the HP that has been removed is replaced, and the new one is removed.

    Args:
        hist (dictionnary): History of various values
        red (dictionnary): Redescription of the feature space
        last (np.array): Values concerning a HP that just has been removed
        d_t (int): Bias associated to the new HP
    """
    hist['non_zero_param'] = last['non_zero_param'].copy()
    red['train'] = last['red_train'].copy()
    red['valid'] = last['red_valid'].copy()
    red['test'] = last['red_test'].copy()
    hist['wb'] = last['wb'].copy()
    hist['cd'] = last['cd'].copy()


def fix_last(hist, red, last):
    """
    Fixes various values in the "last" dictionnary.

    Args:
        hist (dictionnary): History of various values
        red (dictionnary): Redescription of the feature space
        last (np.array): Values concerning a HP that just has been removed
        param (int): Number of non-zero parameters of the current HP
    """
    last['non_zero_param'] = hist['non_zero_param'].copy()
    last['red_train'] = red['train'].copy()
    last['red_valid'] = red['valid'].copy()
    last['red_test'] = red['test'].copy()
    last['wb'] = hist['wb'].copy()
    last['cd'] = hist['cd'].copy()


def obtain_perf(hist, red, dy, curr_num_neur, tr_wei, vd_wei, te_wei, train, valid=None, test=None, c_1 = 0.5, maxx=False):
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
    uniques, pred_reg = None, None
    if maxx:
        tr_err = np.mean(tr_wei * np.sum((train[:, -dy:] - np.maximum(np.reshape(np.matmul(red['train'][:, :curr_num_neur + 2],
                         hist['cd'][:curr_num_neur + 2]), (-1, dy)), 0)) ** 2, axis=1))
    else:
        tr_err = np.mean(tr_wei *
            np.sum((train[:, -dy:] - np.reshape(np.matmul(red['train'][:, :curr_num_neur + 2],
                                                                     hist['cd'][:curr_num_neur + 2]), (-1, dy))) ** 2, axis=1))
    vd_err = np.mean(vd_wei * np.sum((valid[:, -dy:] - np.reshape(np.matmul(
            red['valid'][:, :curr_num_neur + 2], hist['cd'][:curr_num_neur + 2]), (-1, dy))) ** 2, axis=1)) if valid is not None else 0
    te_err = np.mean(te_wei * np.sum((test[:, -dy:] - np.reshape(np.matmul(
            red['test'][:, :curr_num_neur + 2], hist['cd'][:curr_num_neur + 2]), (-1, dy))) ** 2, axis=1)) if test is not None else 0
    return tr_err, vd_err, te_err, uniques, pred_reg


def print_perf(tr_err, vd_err, te_err, tr_var, vd_var, te_var, architecture, hist, cur_num_neur):
    """
    Display the current hidden architecture of the network and the performances

    Args:
        tr_err (float): Training error
        vd_err (float): Validation error
        te_err (float): Test error
        architecture (str): Architecture of the preceding hidden layers of the BNN.
        hist (dictionnary): History of various values
    """
    print('Hid. arch.: ' + architecture + str(cur_num_neur + 1))
    print(f'Train loss: {round(tr_err, 4)} -- Train R2: {round(1 - tr_err / tr_var, 4)}')
    print(f'Valid loss: {round(vd_err, 4)} -- Valid R2: {round(1 - vd_err / vd_var, 4)}')
    print(f'Test  loss: {round(te_err, 4)} -- Test  R2: {round(1 - te_err / te_var, 4)}')
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
    return np.reshape(train[:, -dy:] - np.matmul(red['train_with_bias'], hist['param']), -1, order='F')


def remove_hp(hist, red, train, valid, test, dy, last, dim, task):
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
    tr_err, vd_err, te_err = obtain_perf(hist, red, dy, train, valid, test, task=task)[:3]
    red['train_with_bias'] = red['train_with_bias'][:, dim:]
    red['train'] = red['train'][:, dim:]
    red['valid'] = red['valid'][:, dim:]
    red['test'] = red['test'][:, dim:]
    hist['non_zero_param'] = hist['non_zero_param'][1:]
    hist['param'] = hist['param'][dim:]
    hist['wb'] = hist['wb'][dim:]
    hist['cd'] = hist['cd'][dim:]
    if task == 'regression':
        last['tr_loss'] = tr_err.copy()
        last['non_zero_param'] = hist['non_zero_param'][0].copy()

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
    plt.plot(np.array(range(1, len(hist['tr_loss']) + 1)), np.array(hist['tr_loss']))
    plt.plot(np.array(range(1, len(hist['vd_loss']) + 1)), np.array(hist['vd_loss']))
    plt.plot(np.array(range(1, len(hist['te_loss']) + 1)), np.array(hist['te_loss']))
    plt.legend(['GBN 4.0 - Train',
                'GBN 4.0 - Valid',
                'GBN 4.0 - Test'])
    plt.xlim(0, len(hist['tr_loss']) + 1)
    plt.ylim(0, max(max(hist['tr_loss'][2:]), max(hist['vd_loss'][2:]), max(hist['te_loss'][2:])))
    plt.show()

def is_job_already_done(experiment_name, task_dict, fold_number=None):
    cnt_nw = 0
    try:
        with open("results/" + str(experiment_name) + "_done.txt", "r") as tes:
            tess = [line.strip().split('\t') for line in tes]
        tes.close()
        is_it_new = ['BGN', task_dict['dataset_name'], str(task_dict['seed']), str(task_dict['n_folds'])]
        if fold_number is not None:
            is_it_new += [str(fold_number)]
        is_it_new += [str(task_dict['max_neur']), str(task_dict['max_hdlr']),
                     str(task_dict['max_repl']), str(task_dict['max_coef_per_neur']), str(task_dict['patience'])]
        for a in tess:
            if fold_number is None:
                a.pop(4)
            if a[:-5] == is_it_new:
                cnt_nw += 1
    except FileNotFoundError:
        file = open("results/" + str(experiment_name) + ".txt", "a")
        file.write("algo\tdataset\tseed\tfold\tfold_number\tarchitecture\tparam\tinput\trepl\tmax_coef\tpatience\ttrain_loss\tvalid_loss\ttest_loss\n")
        file.close()
        file = open("results/" + str(experiment_name) + "_done.txt", "a")
        file.write("algo\tdataset\tseed\tfolds\tfold_number\tnb_neur\thid\trepl\tmax_coef\tpatience\ttime\n")
        file.close()
    return cnt_nw


def load_mtpl2(n_samples=None):
    """Fetch the French Motor Third-Party Liability Claims dataset.

    Parameters
    ----------
    n_samples: int, default=None
      number of samples to select (for faster run time). Full dataset has
      678013 samples.
    """
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = fetch_openml(data_id=41215, as_frame=True).data

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    # unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")
    return df.iloc[:n_samples]


def load_dataset(dataset, sep, randseed, folds):
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
    feature_names, dx = None, None
    if dataset == 'bike_day':
        data = pd.read_csv("datasets/bike.sharing.day.csv", sep=",", header=None)
        data = np.array(np.array(data)[1:, 2:-2], dtype=float)
        dy = 1
    elif dataset == 'bike_hour':
        data = pd.read_csv("datasets/bike.sharing.hour.csv", sep=",", header=None)
        data = np.delete(np.array(np.array(data)[1:, 2:], dtype=float), [-2, -3], axis=1)
        dy = 1
    elif dataset == 'carbon':
        data = np.array(pd.read_csv("datasets/carbon_nanotubes.csv", sep=",", header=None))
        dy = 3
    elif dataset == 'diabete':
        data = load_diabetes()
        X = data['data']
        y = data['target']
        data = np.hstack((X, np.reshape(y, (-1, 1))))
        dy = 1
    elif dataset == 'housing':
        data = np.array(pd.read_csv("datasets/housing.csv", sep=",", header=None))
        data[:, -1] /= 100000
        data[:, -2] *= 10
        dy = 1
    elif dataset == 'hung_pox':
        data = np.array(pd.read_csv("datasets/hungary.chickenpox.csv", sep=",", header=None))[1:, 1:]
        for i in range(len(data)):
            data[i, -1] = data[i, -1][0]
        data = np.array(data, dtype=float)
        data = np.hstack((data[:, -3:], data[:, 0:-3]))
        dy = 1
    elif dataset == 'ist_stock_usd':
        data = np.array(pd.read_csv("datasets/istanbul.stock.usd.csv", sep=",", header=None))[:-2]
        data[:, [1, 0]] = data[:, [0, 1]]
        data = np.array(data, dtype=float)
        data[:, 4:] *= 100
        dy = 1
    elif dataset == 'ist_stock_tl':
        data = np.array(pd.read_csv("datasets/istanbul.stock.tl.csv", sep=",", header=None))[:-2]
        data[:, [1, 0]] = data[:, [0, 1]]
        data = np.array(data, dtype=float)
        data[:, 4:] *= 100
        dy = 1
    elif dataset == 'parking':
        data = np.array(pd.read_csv("datasets/parking.birmingham.csv", sep=",", header=None))[1:, :-1]
        data[:, [1, 0]] = data[:, [0, 1]]
        uni = np.unique(data[:, -3]).tolist()
        code = []
        cnt = 0
        for i in range(len(data)):
            if data[i, -3] == uni[0]:
                code.append(cnt)
            else:
                uni.pop(0)
                cnt += 1
                code.append(cnt)
        data = np.hstack((np.reshape(np.array(code), (len(data), 1)), data[:, 1:]))
        data = np.hstack((data[:, :-3], data[:, -2:]))
        data = np.array(data, dtype=float)
        dy = 1
    elif dataset == 'power_plant':
        data = np.array(pd.read_csv("datasets/power_plant.csv", sep=",", header=None))
        dy = 1
    elif dataset == 'solar_flare':
        data = np.array(pd.read_csv("datasets/solar_flare.csv", sep=",", header=None, dtype=float))
        dy = 3
    elif dataset == 'portfolio':
        data = np.array(pd.read_csv("datasets/stock_portfolio.csv", sep=",", header=None))
        dy = 6
    elif dataset == 'turbine':
        data = np.array(pd.read_csv("datasets/turbine.csv", sep=",", header=None))
        dy = 2
    #elif dataset == 'UTKFace':
    #    X = []
    #    y = []
    #    c = 0
    #    for root, dirs, files in os.walk("datasets/UTKFace", topdown=False):
    #        for i in files:
    #            c += 1
    #            # load the image
    #            image = Image.open("datasets/UTKFace/" + i)
    #
    #            # convert image to numpy array
    #            # image = image.resize((32,32, 1))
    #            image = ImageOps.grayscale(image.resize((32, 32)))
    #            # plt.imshow(image)
    #            # plt.show()
    #            X.append(np.asarray(image))
    #            terminator = i.index('_')
    #            y.append(int(i[:terminator]))
    #            # if c % 5000 == 0 :
    #            #    break
    #    X = np.reshape(np.array(X), newshape=(23708, -1))  # 23708
    #    y = np.reshape(np.array(y), newshape=(23708, -1))
    #    data = np.hstack((X, y))
    #    data = data.astype(float, copy=False)
    #    dx = [32, 32, 1]
    #    dy = 1
    elif dataset == 'moons':
        X, y = make_moons(10000, noise=0.1)
        data = np.hstack((X, np.reshape(y, (-1, 1))))
        dy = 1
    elif dataset in ['mnist17', 'mnist56', 'mnist49']:
        low  = dataset[-2]
        high = dataset[-1]
        X_low = np.loadtxt(join("datasets/mnist", f"mnist_{low}")) / 255
        y_low = -1 * np.ones(X_low.shape[0])

        X_high = np.loadtxt(join("datasets/mnist", f"mnist_{high}")) / 255
        y_high = np.ones(X_high.shape[0])

        X = np.vstack((X_low, X_high))
        y = np.hstack((y_low, y_high)).reshape(-1, 1)
        data = np.hstack((X, y))
        data = data.astype(float, copy=False)
        dy = 1
    elif dataset == 'mnistlh':
        X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9 = \
            np.loadtxt(join("datasets/mnist", f"mnist_0")) / 255, \
            np.loadtxt(join("datasets/mnist", f"mnist_1")) / 255, \
            np.loadtxt(join("datasets/mnist", f"mnist_2")) / 255, \
            np.loadtxt(join("datasets/mnist", f"mnist_3")) / 255, \
            np.loadtxt(join("datasets/mnist", f"mnist_4")) / 255, \
            np.loadtxt(join("datasets/mnist", f"mnist_5")) / 255, \
            np.loadtxt(join("datasets/mnist", f"mnist_6")) / 255, \
            np.loadtxt(join("datasets/mnist", f"mnist_7")) / 255, \
            np.loadtxt(join("datasets/mnist", f"mnist_8")) / 255, \
            np.loadtxt(join("datasets/mnist", f"mnist_9")) / 255
        X_low = np.vstack((X_0, X_1, X_2, X_3, X_4))
        X_high = np.vstack((X_5, X_6, X_7, X_8, X_9))
        y_low = -1 * np.ones(X_low.shape[0])
        y_high = np.ones(X_high.shape[0])

        X = np.vstack((X_low, X_high))
        y = np.hstack((y_low, y_high)).reshape(-1, 1)
        data = np.hstack((X, y))
        data = data.astype(float, copy=False)
        dy = 1
    elif 'surpoids' in dataset:
        X = np.array(pd.read_csv("datasets/res." + dataset[9:] + ".relab.tsv", sep="\t", header=None))[:,1:]
        feature_names = X[0]
        y = np.array(pd.read_csv("datasets/res.labels.tsv", sep="\t", header=None))[1:,1:]
        data = np.hstack((X[1:], y))
        data = data.astype(float, copy=False)
        dy = 1
    elif 'covid' in dataset:
        if dataset == 'covid_proteomics':
            h5py_object = h5py.File('datasets/biobanq_full.hdf5', 'r')
            X = np.array(h5py_object['View{}'.format(1)][...])
            feature_names = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(1)][...]]
        elif dataset == 'covid_metabolites':
            h5py_object = h5py.File('datasets/biobanq_full.hdf5', 'r')
            X = np.array(h5py_object['View{}'.format(0)][...])
            feature_names = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(0)][...]]
        elif dataset == 'covid_full_multi_omic':
            h5py_object = h5py.File('datasets/biobanq_full.hdf5', 'r')
            X_1 = np.array(h5py_object['View{}'.format(0)][...])
            X_2 = np.array(h5py_object['View{}'.format(1)][...])
            X = np.hstack((X_1, X_2))
            feature_names = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(0)][...]]+ \
                            [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(1)][...]]

        elif dataset == 'covid_proteomics_signature':
            h5py_object = h5py.File('datasets/biobanq_sign.hdf5', 'r')
            X = np.array(h5py_object['View{}'.format(1)][...])
            feature_names = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(1)][...]]
        elif dataset == 'covid_metabolites_signature':
            h5py_object = h5py.File('datasets/biobanq_sign.hdf5', 'r')
            X = np.array(h5py_object['View{}'.format(0)][...])
            feature_names = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(0)][...]]
        elif dataset == 'covid_both_mono_omic_signature':
            h5py_object = h5py.File('datasets/biobanq_sign.hdf5', 'r')
            X_1 = np.array(h5py_object['View{}'.format(0)][...])
            X_2 = np.array(h5py_object['View{}'.format(1)][...])
            X = np.hstack((X_1, X_2))
            feature_names = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(0)][...]] + \
                            [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(1)][...]]

        elif dataset == 'covid_proteomics_multi_omic_signature':
            h5py_object = h5py.File('datasets/biobanq_multi_sign.hdf5', 'r')
            X = np.array(h5py_object['View{}'.format(1)][...])
            feature_names = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(1)][...]]
        elif dataset == 'covid_metabolites_multi_omics_signature':
            h5py_object = h5py.File('datasets/biobanq_multi_sign.hdf5', 'r')
            X = np.array(h5py_object['View{}'.format(0)][...])
            feature_names = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(0)][...]]
        elif dataset == 'covid_multi_omic_signature':
            h5py_object = h5py.File('datasets/biobanq_multi_sign.hdf5', 'r')
            X_1 = np.array(h5py_object['View{}'.format(0)][...])
            X_2 = np.array(h5py_object['View{}'.format(1)][...])
            X = np.hstack((X_1, X_2))
            feature_names = [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(0)][...]] + \
                            [id.decode() for id in h5py_object["Metadata"]["feature_ids-View{}".format(1)][...]]

        if dataset in ['covid_proteomics', 'covid_metabolites', 'covid_full_multi_omic']:
            h5py_object = h5py.File('datasets/biobanq_full.hdf5', 'r')
        elif dataset in ['covid_proteomics_signature', 'covid_metabolites_signature', 'covid_both_mono_omic_signature']:
            h5py_object = h5py.File('datasets/biobanq_sign.hdf5', 'r')
        elif dataset in ['covid_proteomics_multi_omic_signature', 'covid_metabolites_multi_omics_signature', 'covid_multi_omic_signature']:
            h5py_object = h5py.File('datasets/biobanq_multi_sign.hdf5', 'r')
        y = np.array(h5py_object["Labels"][...])
        data = np.hstack((X, np.reshape(y, (-1,1))))
        data = data.astype(float, copy=False)
        dy = 1
    elif dataset[:6] == "anions":
        if dataset[-2] == '0':
            X, y = get_selected_data(False, float(dataset[-4:]))
        else:
            X, y = get_selected_data(False, float(0))
        data = np.hstack((np.array(X), np.reshape(np.array(y), (-1,1))))
        data = data.astype(float, copy=False)
        dy = 1
        feature_names = X.columns.values.tolist()
    elif dataset[:7] == "cations":
        if dataset[-2] == '0':
            X, y = get_selected_data(True, float(dataset[-4:]))
        else:
            X, y = get_selected_data(True, float(0))
        data = np.hstack((np.array(X), np.reshape(np.array(y), (-1, 1))))
        data = data.astype(float, copy=False)
        dy = 1
        feature_names = X.columns.values.tolist()
    elif "MTPL2" in dataset:
        df = load_mtpl2()

        # Correct for unreasonable observations (that might be data error)
        # and a few exceptionally large claim amounts
        df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
        df["Exposure"] = df["Exposure"].clip(upper=1)
        df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)
        # If the claim amount is 0, then we do not count it as a claim. The loss function
        # used by the severity model needs strictly positive claim amounts. This way
        # frequency and severity are more consistent with each other.
        df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

        log_scale_transformer = make_pipeline(
            FunctionTransformer(func=np.log), StandardScaler()
        )

        column_trans = ColumnTransformer(
            [
                (
                    "binned_numeric",
                    KBinsDiscretizer(n_bins=10, random_state=0),
                    ["VehAge", "DrivAge"],
                ),
                (
                    "onehot_categorical",
                    OneHotEncoder(),
                    ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
                ),
                ("passthrough_numeric", "passthrough", ["BonusMalus"]),
                ("log_scaled_numeric", log_scale_transformer, ["Density"]),
            ],
            remainder="drop",
        )

        # Insurances companies are interested in modeling the Pure Premium, that is
        # the expected total claim amount per unit of exposure for each policyholder
        # in their portfolio:
        df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]

        # This can be indirectly approximated by a 2-step modeling: the product of the
        # Frequency times the average claim amount per claim:
        df["Frequency"] = df["ClaimNb"] / df["Exposure"]
        df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)

        X = column_trans.fit_transform(df)
        X = X.toarray()
        if dataset == "MTPL2_frequency":
            y = np.array(df["Frequency"].to_numpy() > 0)
            weight = df["Exposure"].to_numpy()
        elif dataset == "MTPL2_severity":
            mask = df["ClaimAmount"] > 0
            X = X[mask.values]
            y = df["AvgClaimAmount"][mask.values].to_numpy()
            weight = df["ClaimNb"][mask.values].to_numpy()
        elif dataset == "MTPL2_pure":
            y = df["PurePremium"].to_numpy()
            weight = df["Exposure"].to_numpy()
        data = np.hstack((np.array(X), np.reshape(np.array(weight), (-1, 1))))
        data = np.hstack((data, np.reshape(np.array(y), (-1, 1))))
        data = data.astype(float, copy=False)
        dy = 1

    tr, te = [], []
    if folds == 0:
        args_tr = np.random.choice(len(data), int(len(data) * sep), replace=False)
        args_te = ~np.in1d(range(len(data)), args_tr)
        tr.append(data[args_tr])
        te.append(data[args_te])
    if folds > 0 :
        np.random.shuffle(data)
        kf = KFold(n_splits=folds)
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            tr.append(data[train_index])
            te.append(data[test_index])
    return tr, te, dx, dy, feature_names
