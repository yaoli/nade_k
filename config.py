# -*- coding: utf-8 -*-
from jobman import DD
import utils
exp_path = '/data/lisatmp3/yaoli/exp/nade_k_nips14_release/'
#exp_path = './exps/'
best_1h_model = exp_path + 'reproduce_h500_k5_oldcode_pretrain_finetune/'
best_2h_model = exp_path + 'reproduce_h500_h500_k5_oldcode_pretrain_finetune/'
config = DD({
    'model': 'DeepOrderlessNADE',
    'load_trained': DD({
        # action: 0 standard train, 1 load trained model and evaluate, 2 continue training
        'action': 1,
        'from_path': best_2h_model,
        'epoch': 3999, 
        }),
    'random_seed': 1234,
    'save_model_path': exp_path + '/nade_k_nips14_release_final/test_h2/',
    'dataset': DD({
        'signature': 'MNIST_binary_russ',
        }),
    'DeepOrderlessNADE': DD({
        'n_in': None,
        'n_out': None,
        'n_hidden': 500,
        'n_layers': 2,
        'hidden_act': 'tanh',
        'tied_weights': False,
        # only for the first step of mean field
        'use_mask': False,
        # use data mean to intialize the mean field
        'init_mean_field': True,
        # not avg cost over k steps but only take the cost from the last step
        'cost_from_last': False,
        # 1:0.01 gaussian,2: formula
        'init_weights': 1,
        # centering v
        'center_v': False,
        'train': DD({
            # valid once every 'valid_freq' epochs
            'valid_freq': 250,
            # compute valid and test LL over this many of orderings
            'n_orderings': 5,
            'n_epochs': 1000,
            'minibatch_size': 100,
            # 0 for momentum, 1 for adadelta
            'sgd_type': 1,
            'momentum': 0.9,
            'lr': 0.001,
            # 0.0012279827881 for 2h model
            # 0.0 for 1h model
            'l2': 0.0012279827881,
            # number of mean field steps
            'k': 5,
            'verbose': True,
            'fine_tune': DD({
                'activate': True,
                'n_epochs': 3000,
                })
            })
        })
    })
