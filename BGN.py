# Copyright 2024 Benjamin Leblanc

# This file is part of Binary Greedy Network (BGN).

# BGN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from utils import *
from warnings import simplefilter
from sklearn.model_selection import ParameterGrid
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


def launch(experiment_name='diabete',
           dataset_name=['diabete'],
           seed=[0],
           folds=[0],
           nb_max_neur=[15],
           nb_max_hdlr=[1],
           max_repl_pr_epoch=[100],
           nb_max_coef_per_neur=[2],  # Put to 0 to have standard linear regression
           p=[1],  # [1,4,7],                        # Doesn't matter if regression
           initial_C=[1e3],  # Put to 0 if classification or if doesn't matter
           patience=[15]):
    """
        Given a dataset, iteratively builds a BNN's layer for classification or
            regression purposes using the BGN algorithm. Returns a dataset corresponding
            to the position of the examples in the hyperplane arrangement created
            by the hidden layer.

        Args:
            dataset_name (str): The dataset name. Choices: 'bike_hour', 'carbon', 'diabete', 'housing', 'hung_pox',
                                                           'ist_stock_usd', 'parking', 'power_plant', 'solar_flare',
                                                           'portfolio' or 'turbine'.
            # "anions",
            # "anions_0.02",
            # "anions_0.05",
            # "cations",
            # "cations_0.02",
            # "cations_0.05",

            seed (int): A random seed to use.
            folds (): Number of folds. 0 for standard k for k-CV
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
    param_grid = ParameterGrid([{'dataset_name': dataset_name,
                                 'seed': seed,
                                 'folds': folds,
                                 'nb_max_neur': nb_max_neur,
                                 'nb_max_hdlr': nb_max_hdlr,
                                 'max_repl_pr_epoch': max_repl_pr_epoch,
                                 'nb_max_coef_per_neur': nb_max_coef_per_neur,
                                 'p': p,
                                 'initial_C': initial_C,
                                 'patience': patience}])
    param_grid = [t for t in param_grid]
    ordering = {d: i for i, d in enumerate(dataset_name)}
    param_grid = sorted(param_grid, key=lambda d: ordering[d['dataset_name']])
    n_tasks = len(param_grid)
    for i, task_dict in enumerate(param_grid):
        print(f"Launching task {i + 1}/{n_tasks} : {task_dict}\n")
        cnt_nw = is_job_already_done(experiment_name, task_dict, fold_number=None)
        if (cnt_nw >= max(task_dict['folds'], 1)):
            print("Already done; passing...\n")
        else:
            train_load, test, dx, dy, f_n = load_dataset(task_dict['dataset_name'], 0.80, task_dict['seed'],
                                                         task_dict['folds'])
            for fold_num in range(max(task_dict['folds'], 1)):
                cnt_nw = is_job_already_done(experiment_name, task_dict, fold_number=fold_num)
                if (cnt_nw == 0):
                    if task_dict['folds'] > 0:
                        print(f"\tLaunching fold {fold_num + 1}/{task_dict['folds']}\n")
                    train, valid = train_valid_loaders(train_load[fold_num], train_split=0.75)
                    arch = ''
                    t_before = time()
                    fc_lay = create_layer
                    weighting = True if 'MTPL2' in task_dict['dataset_name'] else False
                    tr_l, vd_l, te_l, nn = fc_lay((train, valid, test[fold_num], f_n),
                                                   task_dict['dataset_name'],
                                                   task_dict['seed'],
                                                   task_dict['folds'],
                                                   fold_num,
                                                   task_dict['nb_max_neur'],
                                                   task_dict['p'],
                                                   arch,
                                                   0,
                                                   task_dict['max_repl_pr_epoch'],
                                                   task_dict['nb_max_coef_per_neur'],
                                                   task_dict['patience'],
                                                   dy,
                                                   experiment_name,
                                                   weighting)
                    num_hid_lay = 1
                    file = open("results/" + str(experiment_name) + "_done.txt", "a")
                    file.write('BGN' + '\t' + task_dict['dataset_name'] + '\t' + str(task_dict['seed']) +
                               '\t' + str(task_dict['folds']) + '\t' + str(fold_num) +
                               '\t' + str(task_dict['nb_max_neur']) + '\t' + str(num_hid_lay) +
                               '\t' + str(task_dict['max_repl_pr_epoch']) + '\t' + str(
                               task_dict['nb_max_coef_per_neur']) + '\t' +
                               str(task_dict['p']) + "\t" + str(task_dict['initial_C']) + "\t" +
                               str(task_dict['patience']) + "\t" + str(tr_l) + "\t" + str(vd_l) + "\t" + str(te_l) +
                               "\t" + str(nn) + "\t" + str(time() - t_before) + "\n")
                    file.close()
launch()
