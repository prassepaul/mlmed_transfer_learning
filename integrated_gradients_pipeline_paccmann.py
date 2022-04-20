# imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json
import sys
from pycombat import Combat

import matplotlib.pyplot as plt
import utils as utils
import argparse
import socket
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
import tensorflow as tf
import models as models
import evaluation as evaluation
import traceback
import paccmann_model as paccmann_model

import argparse
import joblib
import transfer_learning_pipeline as transfer_learning_pipeline
from tensorflow.keras.callbacks import EarlyStopping
import integrated_gradients as ig
    


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def list_string(s):
    split_string = s.split(',')
    out_list = []
    for split_elem in split_string:
        out_list.append(split_elem.strip().replace('\'','').replace('[','').replace(']',''))
    return out_list


def main():
    batch_size = 128
    percentage_tune = 0.1
    early_stopping_patience = 5
    verbose = 1
    use_combat = True
    transform_gene_data = True
    epochs_pretrain = 10
    epochs = 100


    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU','--GPU',type=int,default=0)    
    parser.add_argument('-param_json_path', '--param_json_path', type=str, default='data/params.json')    
    parser.add_argument('-source', '--source', type=str, default='gdsc')    
    parser.add_argument('-target', '--target', type=str, default='beat_aml')    
    parser.add_argument('-use_netpop', '--use_netpop', type=str, default='ensemble')
    parser.add_argument('-seed','--seed', type=int, default = 42)
    parser.add_argument('-flag_normalize_descriptors','--flag_normalize_descriptors',type=str,default='True')
    parser.add_argument('-use_samples','--use_samples',type=int,default=10000)
    parser.add_argument('-train_mode','--train_mode',type=str,default='drug_repurposing')
    parser.add_argument('-save_dir','--save_dir',type=str,default='results_integrated_gradients/')
    parser.add_argument('-save_prefix','--save_prefix',type=str,default='')
    parser.add_argument('-flag_redo','--flag_redo',type=str,default='True')
    parser.add_argument('-flag_pretrain','--flag_pretrain',type=str,default='True')
    
    args = parser.parse_args()
    GPU = args.GPU
    param_json_path = args.param_json_path
    source = args.source
    target = args.target
    use_netpop = args.use_netpop
    seed = args.seed
    flag_normalize_descriptors = boolean_string(args.flag_normalize_descriptors)
    use_samples = args.use_samples
    train_mode = args.train_mode
    save_dir = args.save_dir
    save_prefix = args.save_prefix
    flag_redo = boolean_string(args.flag_redo)
    flag_pretrain = boolean_string(args.flag_pretrain)
    
    save_path = save_dir +  save_prefix + source + '_' + target + '_' + str(use_samples) +\
                '_' + str(flag_normalize_descriptors) + '_' + str(train_mode) + '_' + str(use_netpop) + '_flag_pretrain_' +\
                str(flag_pretrain) + '_paccmann.joblib'
                
    if os.path.exists(save_path) and not flag_redo:
        return
    
    if train_mode == 'drug_repurposing':
        cv_key = None
    elif train_mode == 'precision_oncology':
        cv_key = 'lab_data'
    elif train_mode == 'drug_development':
        cv_key = 'inhib_data'
    
    
    # read params
    with open(param_json_path) as json_file:
        param_dict = json.load(json_file)

    
    # select GPU
    # select graphic card
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    tf_session = tf.compat.v1.Session(config=config)
    
    
    cur_train_pretrain_data_dict = transfer_learning_pipeline.get_train_pretrain_data(param_json_path = param_json_path,
                            source=source,
                            target=target,
                            use_netpop=use_netpop,
                            seed = seed,
                            flag_normalize_descriptors=flag_normalize_descriptors,
                            train_mode=train_mode)
                            
    train_data = cur_train_pretrain_data_dict['train_data']
    pre_train_data = cur_train_pretrain_data_dict['pre_train_data']
    min_max_scaler = cur_train_pretrain_data_dict['min_max_scaler']
    min_max_scaler_descr = cur_train_pretrain_data_dict['min_max_scaler_descr']
    gene_list_dict = cur_train_pretrain_data_dict['gene_list_dict']
    gene_list = cur_train_pretrain_data_dict['gene_list']
    smiles_character_dict = cur_train_pretrain_data_dict['smiles_character_dict']
    gene_list_used = cur_train_pretrain_data_dict['gene_list_used']
    
    character_smiles_dict = dict()
    for smiles in smiles_character_dict:
        character_smiles_dict[smiles_character_dict[smiles]] = smiles
    
    # get model_params
    result_path = param_dict['model_param_pretrain_csv']

    #use_netpop = None
    model_params, gene_key, gene_use_ids = models.get_best_model_params(result_path,
                         gene_list_dict = gene_list_dict,
                         complete_gene_list = gene_list,
                         gene_list = use_netpop)

    # paccmann params
    model_params.update({
      "batch_size": 64,
      "decay_rate": 0.96,
      "decay_steps": 3000,
      "dropout": 0.3,
      "eval_batch_size": 32,
      "filter": [64,64,64],
      "genes_number": model_params['num_gene_features'],
      "kernels": [[3,16], [5,16], [11, 16]],
      "learning_rate": 0.0002,
      "max_num_epochs": 200,
      "multiheads": [4,4,4,4],
      "patience": 15,
      "smiles_attention_size": 64,
      "smiles_embedding_size": 16,
      "smiles_length": model_params['drug_len'],
      "smiles_vocab": model_params['vocab_size'],
      "stacked_dense_hidden_sizes": [512, 128, 64, 16],
    })


    # rf params
    model_params.update({'num_trees':100})

    epochs_pretrain = {'nn_baseline':10,
                       'nn_paccmann':100,
                       'rf':None,
                       'tDNN':10}

    # tDNN
    model_params.update({'drug_descriptors':train_data['drug_data_des'].shape[1]})
        
    num_gene_features = model_params['num_gene_features']    
    drug_len = model_params['drug_len']
    embed_dim = model_params['embed_dim']
    drug_filters = model_params['drug_filters']
    drug_kernels = model_params['drug_kernels']
    pool_sizes = model_params['pool_sizes']
    dense_layers = model_params['dense_layers']
    use_batch_decorrelation = model_params['use_batch_decorrelation']
    use_normal_batch_norm = model_params['use_normal_batch_norm']

    vocab_size = model_params['vocab_size']

    gene_data       = train_data['gene_data']
    if transform_gene_data:
            gene_data   = utils.transformation_np(gene_data)
    drug_data       = train_data['drug_data']
    drug_data_des   = train_data['drug_data_des']
    label           = train_data['label']
    inhib_data      = train_data['inhib_data']
    lab_data        = train_data['lab_data']

    # transform input
    drug_data[drug_data >= vocab_size] = 0
     

    train_label         = label
    train_gene_data     = gene_data
    train_drug_data     = drug_data
    train_drug_des_data = drug_data_des

    # select tuning data
    np.random.seed(seed)
    rand_idx = np.random.permutation(np.arange(train_gene_data.shape[0]))
    end_tune = int(np.ceil(percentage_tune * len(rand_idx)))
    tune_ids = rand_idx[0:end_tune]
    train_ids = rand_idx[end_tune:]
    tune_gene_data = train_gene_data[tune_ids,:]
    tune_drug_data = train_drug_data[tune_ids,:]
    tune_drug_des_data = train_drug_des_data[tune_ids,:]
    tune_label = train_label[tune_ids]
    train_gene_data = train_gene_data[train_ids,:]
    train_drug_data = train_drug_data[train_ids,:]
    train_drug_des_data = train_drug_des_data[train_ids,:]
    train_label = train_label[train_ids]
    train_batch = [0]*len(train_label)
    tune_batch = [0] * len(tune_label)
    
    # perform combat
    from pycombat import Combat
    if pre_train_data is not None:
        gene_data_pre = pre_train_data['gene_data']
        if transform_gene_data:
            gene_data_pre = utils.transformation_np(gene_data_pre)
        drug_data_pre = pre_train_data['drug_data']
        drug_data_des_pre = pre_train_data['drug_data_des']
        label_pre     = pre_train_data['label']
        train_batch += [1] * len(label_pre)

        if use_combat:
            combat = Combat()
            complete_combat_data = np.vstack([train_gene_data,gene_data_pre])
            combat.fit(Y = complete_combat_data, b = train_batch, X = None, C = None)

            gene_data_pre = combat.transform(Y = np.vstack([gene_data_pre,complete_combat_data]), b = [1] * len(label_pre) + train_batch, X = None, C = None)
            gene_data_pre = gene_data_pre[0:len(label_pre),:]
            train_gene_data = combat.transform(Y = np.vstack([train_gene_data,complete_combat_data]), b = [0] * len(train_label) + train_batch, X = None, C = None)
            train_gene_data = train_gene_data[0:len(train_label),:]
            tune_gene_data = combat.transform(Y = np.vstack([tune_gene_data,complete_combat_data]), b = [0] * len(tune_label) + train_batch, X = None, C = None) 
            tune_gene_data = tune_gene_data[0:len(tune_label),:]
            # transform input
            drug_data_pre[drug_data_pre >= vocab_size] = 0     
    
    use_epochs_pretrain = epochs_pretrain['nn_paccmann']
    
    
    tf.keras.backend.clear_session()

    model = paccmann_model.get_paccmann_model(model_params)

    model_emb = paccmann_model.get_paccmann_model(model_params,
                                                 flag_embedding_as_input = True)
    
    ###############################################
    #
    # PRETRAIN ON SOURCE
    #
    ###############################################
    if flag_pretrain:
        # pretrain on source data
        model.fit([drug_data_pre,np.zeros([gene_data_pre.shape[0],1]),gene_data_pre],label_pre,epochs = use_epochs_pretrain,
                                        batch_size = batch_size,verbose = verbose)
    
    
    if use_samples is not None:
        rand_idx = np.random.permutation(np.arange(train_gene_data.shape[0]))
        train = rand_idx[0:use_samples]
        train_label = label[train]
        train_gene_data = gene_data[train,:]
        train_drug_data = drug_data[train,:]
    
    ###############################################
    #
    # TRAIN ON TARGET
    #
    ###############################################
    
    # train on target data
    early_stopping = EarlyStopping(monitor='loss', patience=early_stopping_patience)
    model.fit([train_drug_data,np.zeros([train_drug_data.shape[0],1]),train_gene_data],train_label,epochs = epochs,
                     validation_data=([tune_drug_data,np.zeros([tune_drug_data.shape[0],1]),tune_gene_data],tune_label),
                     batch_size = batch_size,
                     callbacks = [early_stopping],
                     shuffle=True, verbose = verbose)
    
    model_pre, model_con, model_emb = ig.get_sub_models(model,model_emb)
    
    # number of steps to interpolate
    m_steps=50
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    
    gene_data       = train_data['gene_data']
    if transform_gene_data:
            gene_data   = utils.transformation_np(gene_data)
    drug_data       = train_data['drug_data']
    drug_data_des   = train_data['drug_data_des']
    label           = train_data['label']
    inhib_data      = train_data['inhib_data']
    lab_data        = train_data['lab_data']

    predictions = model.predict([drug_data,
                             np.zeros([drug_data.shape[0],1]),
                             gene_data],batch_size = batch_size)
    
    gene_importances = np.zeros(gene_data.shape)
    drug_importances = np.zeros(drug_data.shape)
    for i in tqdm(np.arange(gene_data.shape[0])):
        cur_gene_data = gene_data[i:i+1]
        cur_drug_data = drug_data[i:i+1]
        cur_zero_data = np.zeros([cur_drug_data.shape[0],1])
        cur_data = [cur_drug_data,
                   cur_zero_data,
                   cur_gene_data]
        gene_importances_scaled, drug_importances_scaled = ig.get_gene_drug_importances_for_instance_paccmann(cur_data, model_con,
                                             model_emb, alphas)
        gene_importances[i,:] = gene_importances_scaled
        drug_importances[i,:] = drug_importances_scaled
        
    
    
    joblib.dump({'gene_data':gene_data,
                 'drug_data':drug_data,
                 'drug_data_des':drug_data_des,
                 'label':label,
                 'inhib_data':inhib_data,
                 'lab_data':lab_data,
                 'predictions':predictions,
                 'gene_importances':gene_importances,
                 'drug_importances':drug_importances,
                 'gene_list':gene_list,
                 'gene_list_used':gene_list_used,
                 }, save_path, compress=3, protocol=2)

if __name__ == "__main__":
    # execute only if run as a script
    main() 