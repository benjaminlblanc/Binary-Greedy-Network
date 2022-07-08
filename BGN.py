
# Copyright 2022 Benjamin Leblanc

# This file is part of Binary Greedy Network (BGN).

# BGN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
from time import time
from sklearn.linear_model import Lasso
from sklearn.model_selection import ParameterGrid
from utils import train_valid_loaders, write, find_threshold, expand_dataset_dy, \
    optimal_cd, normalize_input_space, reset_to_last, fix_last, obtain_perf, \
    print_perf, update_y_t, remove_hp, display_evolution, load_dataset

def linear_layer(dataset, 
               dataset_name, 
               seed, 
               max_nb_neur, 
               architecture, 
               res_param,
               max_repl_pr_epoch, 
               dy,
               file_name) :
    """
    Given a dataset, iteratively builds a BNN's layer for classification or 
        regression purposes using the BGN algorithm. Returns a dataset corresponding 
        to the position of the examples in the hyperplane arrangement created 
        by the hidden layer.

    Args:
        dataset (Tuple of n_examples x n_features np.arrays): Train, validation and test datasets.
        dataset_name (str): The dataset name.
        seed (int): A random seed to use.
        max_nb_neur (int): Maximum number of neurons in the layer.
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
    train, valid, test = dataset[0], dataset[1], dataset[2]
    
    # Expand the dataset according to the dimension of the label space (see the BGN paper, Improvement 3)
    exp_train = expand_dataset_dy(train.copy(), dy)    
    X_t = exp_train[:,:-1].copy()
    y_t = exp_train[:,-1].copy()
    
    C = 1e5                             # Regularization parameter    
    hist = {'param':np.zeros((1,dy)),   # History of the parameter values of the output layer
            'non_zero_param':[],        # ... number of non-zero parameters
            'tr_loss':[],               # ... training loss
            'vd_loss':[],               # ... validation loss
            'te_loss':[]}               # ... test loss
    red = {'train_with_bias':[],        # Redescription of the training feature space (with bias)
           'train':[],                  # ... training feature space
           'valid':[],                  # ... validation feature space
           'test':[]}                   # ... test feature space
    last = {'param':np.zeros((1,dy)),   # Parameter values of the output layer before the removal of a HP
            'red_train_with_bias':[],   # Redescription (with bias) of the training feature space ...
            'red_train':[],             # Redescription of the training feature space ...
            'red_valid':[],             # Redescription of the validation feature space ...
            'red_test':[],              # Redescription of the test feature space ...
            'tr_loss':1e100,            # Training loss obtained ...
            'non_zero_param':0}         # Helps keep track of how many non-zero params that have been removed during a remove-replace iteration    
    cnt = 0                             # Keeps track of how many remove-replace iterations have been done
    curr_num_neur = 0                   # Keeps track of how many hyperplanes have been placed    
    best_vd_loss = 1e100
    w_t = [0]
    lasso_reg = Lasso(alpha = C)
    
    while curr_num_neur < max_nb_neur :
        # We use the Lasso regression to find the orientation of the current HP.
        w_t = lasso_reg.fit(X = X_t, y = y_t).coef_
        # If theregularization is too strong, it is minimized a little.
        while np.sum(abs(w_t)) < 1e-5 :
            C /= 1.5
            lasso_reg = Lasso(alpha = C)
            w_t = lasso_reg.fit(X = X_t, y = y_t).coef_
        
        # Find the optimal bias of the current HP.
        b_t = -find_threshold(y_t, np.sum(X_t * w_t, axis = 1))
        
        # Keeps track of the number of non-zero parameters of the current HP.
        param = sum(abs(w_t) > 0)
        
        # Every red_ numpy array corresponds to the output of the hidden layer we are currently building.
        #   It thus corresponds to the redescription of X.        
        if curr_num_neur == 0 :
            red['train'] = np.sign(np.sum(train[:,0:-dy]*w_t, axis = 1)+b_t)
            red['train'] = np.reshape(red['train'], ((-1,1)))
            red['valid'] = np.sign(np.sum(valid[:,0:-dy]*w_t, axis = 1)+b_t)
            red['valid'] = np.reshape(red['valid'], ((-1,1)))
            red['test'] = np.sign(np.sum(test[:,0:-dy]*w_t, axis = 1)+b_t)
            red['test'] = np.reshape(red['test'], ((-1,1)))
        else :
            red['train'] = np.hstack((red['train'], np.reshape(np.sign(np.sum(train[:,0:-dy]*w_t, axis = 1)+b_t), ((-1,1)))))
            red['valid'] = np.hstack((red['valid'], np.reshape(np.sign(np.sum(valid[:,0:-dy]*w_t, axis = 1)+b_t), ((-1,1)))))
            red['test'] = np.hstack((red['test'], np.reshape(np.sign(np.sum(test[:,0:-dy]*w_t, axis = 1)+b_t), ((-1,1)))))
        red['train_with_bias'] = np.hstack((red['train'], np.reshape(np.ones(len(red['train'])), (-1,1))))
        
        # Here, we find the optimal values for c_t and d_t.
        c_t, d_t = optimal_cd(np.reshape(y_t, (-1,dy), order='F'), red['train'], dy)
        hist['param'][-1,:] += d_t.copy()
        hist['param'] = np.vstack((hist['param'][:-1,:],c_t,hist['param'][-1,:]))
        
        # The "if" is respected if a remove-replace iteration is not possible or relevant anymore.
        #   Then, the oldest HP won't be removed.
        if cnt == len(hist['param'])-1 or cnt > max_repl_pr_epoch :
            # Since an iteration of the algorithm is completed, the counter is reset.
            cnt = 0
            
            # If the last remove-replace iteration was successful, than nothing changes;
            #   if the error has not diminished, than the last HP is replaced.
            if last['tr_loss'] < obtain_perf(hist, red, dy, train)[0] + 1e-10 :    
                reset_to_last(hist, red, last, d_t)
            
            # We fix the parameters of the current neural network in the <last> dictionnary
            fix_last(hist, red, last, param)
            
            # We update y_t according to the new HP predictions
            y_t = update_y_t(hist, red, train, dy)
            
            # Performances are obtained, displayed, written and consigned
            tr_err, vd_err, te_err = obtain_perf(hist, red, dy, train, valid, test)
            print_perf(tr_err, vd_err, te_err, architecture, hist)
            write(file_name, 'BGN', dataset_name, seed, architecture+str(len(hist['param'])-1), sum(hist['non_zero_param'])+len(hist['param'])-1+res_param, max_repl_pr_epoch, C, [tr_err, vd_err, te_err])
            hist['tr_loss'].append(tr_err.copy())
            hist['vd_loss'].append(vd_err.copy())
            hist['te_loss'].append(te_err.copy())
            
            # A neuron is added
            curr_num_neur += 1
            last['tr_loss'] = 1e100
            
        # The "if" would have been respected if a remove-replace iteration was not possible anymore.
        #   Therefore, entering the "else" means the oldest HP will be removed.
        else :
            # If the last remove-replace iteration was successful, than nothing changes;
            #   if the error has not diminished, than the last HP is replaced.
            if last['tr_loss'] < obtain_perf(hist, red, dy, train)[0] + 1e-10 \
                and sum(np.abs(last['param'][0])) > 0 :
                reset_to_last(hist, red, last, d_t)
                cnt += 1
            else :
                # If the last remove-replace iteration was successul, than the counter is reset.
                cnt = 0
            # We fix the parameters of the current neural network in the "last" dictionnary
            fix_last(hist, red, last, param)
            
            # We update y_t according to the new HP predictions
            y_t = update_y_t(hist, red, train, dy)
            
            # Performances are obtained
            tr_err, vd_err, te_err = obtain_perf(hist, red, dy, train, valid, test)
            
            # An iteration of the remove-replace technique begins by removing the olest HP
            remove_hp(hist, red, train, valid, test, dy, last, y_t)
            
            # We update y_t once again, since a HP has been removed
            y_t = update_y_t(hist, red, train, dy)
        
        if curr_num_neur > 0 :
            if best_vd_loss > min(hist['vd_loss']) :
                best_vd_loss = min(hist['vd_loss']).copy()
                best_red = red.copy()
                best_param = sum(hist['non_zero_param'].copy())+res_param
            
        if curr_num_neur > 25 :
            if min(hist['vd_loss'][-25:]) > min(hist['vd_loss']) : 
                curr_num_neur = max_nb_neur
                
    # The evolution of the performances in training, validation and test are displayed
    display_evolution(hist)
    
    return np.hstack((best_red['train'], train[:,-dy:])), \
           np.hstack((best_red['valid'], valid[:,-dy:])), \
           np.hstack((best_red['test'],  test[:,-dy:])), \
           best_vd_loss, \
           best_param
           
def launch(experiment_name = 'BGN',
           dataset = ['bike_hour',
                      'carbon',
                      'diabete',
                      'housing',
                      'hung_pox',
                      'ist_stock_usd',
                      'parking',
                      'power_plant',
                      'solar_flare',
                      'portfolio',
                      'turbine'
                      ],
           seed = [0,1,2,3,4],
           nb_max_neur = [1000],
           nb_max_hdlr = [10],
           max_repl_pr_epoch = [100]) :
    param_grid = ParameterGrid([{'dataset': dataset,
                                 'seed': seed,
                                 'nb_max_neur': nb_max_neur,
                                 'nb_max_hdlr': nb_max_hdlr,
                                 'max_repl_pr_epoch': max_repl_pr_epoch}])
    param_grid = [t for t in param_grid]
    ordering = {d: i for i, d in enumerate(dataset)}
    param_grid = sorted(param_grid, key=lambda d: ordering[d['dataset']])
    n_tasks = len(param_grid)
    cnt = 0
    for i, task_dict in enumerate(param_grid):
        print(f"Launching task {i + 1}/{n_tasks} : {task_dict}\n")
        cnt += 1
        cnt_nw = 0
        try:
            with open("results/"+str(experiment_name)+"_done.txt", "r") as tes :
                tess = [line.strip().split('\t') for line in tes]
            tes.close()
            is_it_new = ['BGN', task_dict['dataset'], str(task_dict['seed']), 
                         str(task_dict['nb_max_neur']), str(task_dict['nb_max_hdlr']), 
                         str(task_dict['max_repl_pr_epoch'])]
            for a in tess:
                if a[:-1] == is_it_new :
                    cnt_nw += 1
        except FileNotFoundError :
            file = open("results/"+str(experiment_name)+".txt","a")
            file.write("algo\tdataset\tseed\tarchitecture\tnb_param\tnb_repl\tC\ttrain_loss\tvalid_loss\ttest_loss\n")
            file.close()
            file = open("results/"+str(experiment_name)+"_done.txt","a")
            file.write("algo\tdataset\tseed\tnb_neur\tnb_hid\tnb_repl\ttime\n")
            file.close()
        
        if(cnt_nw >= 1) :
            print("Already done; passing...\n")
        else :
            train_load, test, dy = load_dataset(task_dict['dataset'], 0.8, task_dict['seed'])
            
            train, valid = train_valid_loaders(train_load)
            # It's been found that normalized feature space yields better results
            normalize_input_space(train, valid, test, dy)
            arch = ''
            t_before = time()
            trmod, vamod, temod, min_vd, res_param = linear_layer((train, valid, test),
                                                     task_dict['dataset'],
                                                     task_dict['seed'],
                                                     task_dict['nb_max_neur'], 
                                                     arch, 
                                                     0,
                                                     task_dict['max_repl_pr_epoch'],
                                                     dy,
                                                     experiment_name)
            num_hid_lay = 1
            file = open("results/"+str(experiment_name)+"_done.txt","a")
            file.write('BGN'+'\t'+task_dict['dataset']+'\t'+str(task_dict['seed'])+
                   '\t'+str(task_dict['nb_max_neur'])+'\t'+str(num_hid_lay)+
                   '\t'+str(task_dict['max_repl_pr_epoch'])+"\t"+str(time()-t_before)+"\n")
            file.close()
            minn_vd = min_vd
            while num_hid_lay != task_dict['nb_max_hdlr'] :
                num_hid_lay += 1
                arch += str(len(trmod[0])-1)+'-'
                trmod, vamod, temod, min_vd, res_param = linear_layer((trmod, vamod, temod),
                                                         task_dict['dataset'],
                                                         task_dict['seed'],
                                                         task_dict['nb_max_neur'], 
                                                         arch, 
                                                         res_param,
                                                         task_dict['max_repl_pr_epoch'],
                                                         dy,
                                                         experiment_name)
                file = open("results/"+str(experiment_name)+"_done.txt","a")
                file.write('BGN'+'\t'+task_dict['dataset']+'\t'+str(task_dict['seed'])+
                       '\t'+str(task_dict['nb_max_neur'])+'\t'+str(num_hid_lay)+
                       '\t'+str(task_dict['max_repl_pr_epoch'])+"\t"+str(time()-t_before)+"\n")
                file.close()
                if minn_vd < min_vd :
                    num_hid_lay = task_dict['nb_max_hdlr']
                else : 
                    minn_vd = min_vd
launch()
