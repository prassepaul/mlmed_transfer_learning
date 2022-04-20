import models as models
import paccmann_model as paccmann_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from pycombat import Combat
import utils as utils
        


def get_metric_dict(predictions,label,inhib,lab,
                    min_max_scaler, invers_scale_list = [True,False],
                    flag_calc_per_inhib = True,
                    flag_calc_per_lab = True):
    metric_dict = dict()
    for inverse_scale in invers_scale_list:
        sps  = []
        sps_inhib = []
        sps_lab = []
        r2s  = []
        pes  = []
        pes_inhib = []
        pes_lab = []
        mses = []
        mses_inhib = []
        mses_lab = []
        appendix = ''
        if inverse_scale:
            appendix += '_inverse'
        for i in range(len(label)):
            gt_values = np.array(label[i])
            pred_values = np.array(predictions[i])
            if inhib is not None:
                labs = np.array(lab[i])
                inhibs = np.array(inhib[i])
            else:
                labs = []
                inhibs = []
                
            if inverse_scale and min_max_scaler is not None:
                gt_values = min_max_scaler.inverse_transform(gt_values.reshape(-1, 1))
                pred_values = min_max_scaler.inverse_transform(pred_values.reshape(-1, 1))
            sp=float(pd.DataFrame(gt_values).corrwith(pd.DataFrame(pred_values), drop=True,method='spearman' , axis=0))
            pe=float(pd.DataFrame(gt_values).corrwith(pd.DataFrame(pred_values), drop=True,method='pearson' , axis=0))
            r2 = r2_score(gt_values, pred_values)
            mse = mean_squared_error(gt_values, pred_values)
            #print('pearson: ' + str(pe))
            #print('spearman: ' + str(sp))
            #print('R2: ' + str(r2))
            pes.append(pe)
            r2s.append(r2)
            sps.append(sp)
            mses.append(mse)
            # get the per inhibitor scores
            inhibitors = list(np.unique(inhibs))
            #print('number of inhibitors: ' + str(len(inhibitors)))
            tmp_pes   = []
            tmp_sps   = []
            tmp_meses = []
            if flag_calc_per_inhib:
                for j in range(len(inhibitors)):
                    cur_inhibitor = inhibitors[j]
                    cur_ids = list(np.where(inhibs == cur_inhibitor)[0])
                    #print(cur_inhibitor + ' ' + str(len(cur_ids)))
                    if len(cur_ids) > 0:
                        #print(type(cur_ids))
                        sp_inh=float(pd.DataFrame(gt_values[cur_ids]).corrwith(pd.DataFrame(pred_values[cur_ids]), drop=True,method='spearman' , axis=0))
                        pe_inh=float(pd.DataFrame(gt_values[cur_ids]).corrwith(pd.DataFrame(pred_values[cur_ids]), drop=True,method='pearson' , axis=0))
                        mse_inh = mean_squared_error(gt_values[cur_ids], pred_values[cur_ids])
                        tmp_pes.append(pe_inh)
                        tmp_sps.append(sp_inh)
                        tmp_meses.append(mse_inh)
                if len(tmp_pes) > 0:
                    sps_inhib.append(np.mean(tmp_sps))
                    pes_inhib.append(np.mean(tmp_pes))
                    mses_inhib.append(np.mean(mse_inh))
                else:
                    sps_inhib.append(0)
                    pes_inhib.append(0)
                    mses_inhib.append(0)
            
            # get the per cellline scores
            lab_list = list(np.unique(labs))
            #print('number of labs: ' + str(len(labs)))
            tmp_pes   = []
            tmp_sps   = []
            tmp_meses = []
            if flag_calc_per_lab:
                for j in range(len(lab_list)):
                    cur_lab = lab_list[j]
                    cur_ids = list(np.where(labs == cur_lab)[0])
                    #print(cur_lab + ' ' + str(len(cur_ids)))
                    if len(cur_ids) > 0:
                        #print(type(cur_ids))
                        sp_inh=float(pd.DataFrame(gt_values[cur_ids]).corrwith(pd.DataFrame(pred_values[cur_ids]), drop=True,method='spearman' , axis=0))
                        pe_inh=float(pd.DataFrame(gt_values[cur_ids]).corrwith(pd.DataFrame(pred_values[cur_ids]), drop=True,method='pearson' , axis=0))
                        mse_inh = mean_squared_error(gt_values[cur_ids], pred_values[cur_ids])
                        tmp_pes.append(pe_inh)
                        tmp_sps.append(sp_inh)
                        tmp_meses.append(mse_inh)
                if len(tmp_pes) > 0:
                    sps_lab.append(np.nanmean(tmp_sps))
                    pes_lab.append(np.nanmean(tmp_pes))
                    mses_lab.append(np.nanmean(mse_inh))
                else:
                    sps_lab.append(0)
                    pes_lab.append(0)
                    mses_lab.append(0)
            
        metric_dict['pearson' + appendix] = np.nanmean(pes)
        metric_dict['pearson_std' + appendix] = np.std(pes)
        metric_dict['pearson_list' + appendix] = pes
        metric_dict['spearman' + appendix] = np.nanmean(sps)
        metric_dict['spearman_std' + appendix] = np.std(sps)
        metric_dict['spearman_list' + appendix] = sps
        metric_dict['pearson_inhib' + appendix] = np.nanmean(pes_inhib)
        metric_dict['pearson_inhib_std' + appendix] = np.std(pes_inhib)
        metric_dict['pearson_inhib_list' + appendix] = pes_inhib
        metric_dict['pearson_lab' + appendix] = np.nanmean(pes_lab)
        metric_dict['pearson_lab_std' + appendix] = np.std(pes_lab)
        metric_dict['pearson_lab_list' + appendix] = pes_lab
        metric_dict['spearman_inhib' + appendix] = np.nanmean(sps_inhib)
        metric_dict['spearman_inhib_std' + appendix] = np.std(sps_inhib)
        metric_dict['spearman_inhib_list' + appendix] = sps_inhib
        metric_dict['spearman_lab' + appendix] = np.nanmean(sps_lab)
        metric_dict['spearman_lab_std' + appendix] = np.std(sps_lab)
        metric_dict['spearman_lab_list' + appendix] = sps_lab
        metric_dict['R2' + appendix] = np.nanmean(r2s)
        metric_dict['R2_std' + appendix] = np.std(r2s)
        metric_dict['R2_list' + appendix] = r2s
        metric_dict['MSE' + appendix] = np.nanmean(mses)
        metric_dict['MSE_std' + appendix] = np.std(mses)
        metric_dict['MSE_list' + appendix] = mses
        metric_dict['MSE_inhib' + appendix] = np.nanmean(mses_inhib)
        metric_dict['MSE_inhib_std' + appendix] = np.std(mses_inhib)
        metric_dict['MSE_inhib_list' + appendix] = mses_inhib
        metric_dict['MSE_lab' + appendix] = np.nanmean(mses_lab)
        metric_dict['MSE_lab_std' + appendix] = np.std(mses_lab)
        metric_dict['MSE_lab_list' + appendix] = mses_lab
    return metric_dict




def get_cv_result(n_splits,train_data,pre_train_data=None,
                 model_params = dict(),epochs_pretrain = 10, epochs = 100,
                 cv_key = 'lab_data',
                 batch_size = 128,
                 percentage_tune = 0.1,
                 early_stopping_patience = 5,
                 verbose = 1, num_use_train = None,
                 use_combat = False, transform_gene_data = False):
    if 'num_trees' in model_params:
        model_type = 'rf'
        num_trees = model_params['num_trees']
        from sklearn.ensemble import RandomForestRegressor
    elif 'multiheads' in model_params:
        model_type = 'paccmann'
    else:
        model_type = 'nn'
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
    
    gene_data  = train_data['gene_data']
    if transform_gene_data:
        gene_data = utils.transformation_np(gene_data)
    drug_data  = train_data['drug_data']
    label      = train_data['label']
    inhib_data = train_data['inhib_data']
    lab_data   = train_data['lab_data']
    
    # transform input
    drug_data[drug_data >= vocab_size] = 0
    
    cv_out = GroupKFold(n_splits=n_splits)
    if cv_key is not None:
        cv_out_splits = cv_out.split(gene_data, label, train_data[cv_key])
    else:
        cv_out_splits = cv_out.split(gene_data, label, np.arange(len(label)))
    counter = 1
    gt_complete = []
    pred_complete = []
    inhib_data_complete = []
    lab_data_complete = []
    
    for train, test in tqdm(cv_out_splits,total = n_splits):
        # select only subset of train data
        # TODO: select according to sheme
        if num_use_train is not None:
            rand_idx = np.random.permutation(np.arange(len(train)))
            train = np.array(train)[rand_idx[0:num_use_train]]            
            
        train_label = label[train]
        train_gene_data = gene_data[train,:]
        train_drug_data = drug_data[train,:]
        
        
        
        # select tuning data
        np.random.seed(counter)
        rand_idx = np.random.permutation(np.arange(train_gene_data.shape[0]))
        end_tune = int(np.ceil(percentage_tune * len(rand_idx)))
        tune_ids = rand_idx[0:end_tune]
        train_ids = rand_idx[end_tune:]
        tune_gene_data = train_gene_data[tune_ids,:]
        tune_drug_data = train_drug_data[tune_ids,:]
        tune_label = train_label[tune_ids]
        train_gene_data = train_gene_data[train_ids,:]
        train_drug_data = train_drug_data[train_ids,:]
        train_label = train_label[train_ids]
        train_batch = [0]*len(train_label)
        tune_batch = [0] * len(tune_label)
        
        val_label = label[test]
        val_gene_data = gene_data[test,:]
        val_drug_data = drug_data[test,:]
        val_inhib = inhib_data[test]
        val_lab = lab_data[test]
        test_batch = [0] * len(test)
        
        
        if model_type == 'nn':
            tf.keras.backend.clear_session()
            model = models.get_baseline_nn_model(num_gene_features = num_gene_features,
                vocab_size = vocab_size,
                drug_len = drug_len,
                embed_dim = embed_dim,
                drug_filters = drug_filters,
                drug_kernels = drug_kernels,
                dense_layers = dense_layers,
                pool_sizes = pool_sizes,
                use_batch_decorrelation = use_batch_decorrelation,
                use_normal_batch_norm = use_normal_batch_norm)
            
            # check for pretrain
            if pre_train_data is not None:
                gene_data_pre = pre_train_data['gene_data']
                if transform_gene_data:
                    gene_data_pre = utils.transformation_np(gene_data_pre)
                drug_data_pre = pre_train_data['drug_data']
                label_pre     = pre_train_data['label']
                train_batch += [1] * len(label_pre)
                
                if use_combat:
                    combat = Combat()
                    #print('train_gene_data.shape: ' + str(train_gene_data.shape))
                    #print('gene_data_pre.shape: ' + str(gene_data_pre.shape))
                    #print('np.vstack([train_gene_data,gene_data_pre]).shape: ' + str(np.vstack([train_gene_data,gene_data_pre]).shape))
                    #print('len(train_batch)): ' + str(len(train_batch)))
                                        
                    complete_combat_data = np.vstack([train_gene_data,gene_data_pre])
                    combat.fit(Y = complete_combat_data, b = train_batch, X = None, C = None)
                    
                    gene_data_pre = combat.transform(Y = np.vstack([gene_data_pre,complete_combat_data]), b = [1] * len(label_pre) + train_batch, X = None, C = None)
                    gene_data_pre = gene_data_pre[0:len(label_pre),:]
                    train_gene_data = combat.transform(Y = np.vstack([train_gene_data,complete_combat_data]), b = [0] * len(train_label) + train_batch, X = None, C = None)
                    train_gene_data = train_gene_data[0:len(train_label),:]
                    val_gene_data = combat.transform(Y = np.vstack([val_gene_data,complete_combat_data]), b = [0] * len(val_lab) + train_batch, X = None, C = None)
                    val_gene_data = val_gene_data[0:len(val_lab),:]
                    tune_gene_data = combat.transform(Y = np.vstack([tune_gene_data,complete_combat_data]), b = [0] * len(tune_label) + train_batch, X = None, C = None) 
                    tune_gene_data = tune_gene_data[0:len(tune_label),:]
                    
                # transform input
                drug_data_pre[drug_data_pre >= vocab_size] = 0            
                model.fit([gene_data_pre,drug_data_pre],label_pre,epochs = epochs_pretrain,
                                batch_size = batch_size,verbose = verbose)
            
            early_stopping = EarlyStopping(monitor='loss', patience=early_stopping_patience)
            model.fit([train_gene_data,train_drug_data],train_label,epochs = epochs,
                                 validation_data=([tune_gene_data,tune_drug_data],tune_label),
                                 batch_size = batch_size,
                                 callbacks = [early_stopping],
                                 shuffle=True, verbose = verbose)

            predictions = model.predict([val_gene_data,val_drug_data])
        elif model_type == 'paccmann':
            tf.keras.backend.clear_session()
            model = paccmann_model.get_paccmann_model(model_params)
            
            # check for pretrain
            if pre_train_data is not None:
                gene_data_pre = pre_train_data['gene_data']
                if transform_gene_data:
                    gene_data_pre = utils.transformation_np(gene_data_pre)
                drug_data_pre = pre_train_data['drug_data']
                label_pre     = pre_train_data['label']
                train_batch += [1] * len(label_pre)
                
                if use_combat:
                    combat = Combat()
                    #print('train_gene_data.shape: ' + str(train_gene_data.shape))
                    #print('gene_data_pre.shape: ' + str(gene_data_pre.shape))
                    #print('np.vstack([train_gene_data,gene_data_pre]).shape: ' + str(np.vstack([train_gene_data,gene_data_pre]).shape))
                    #print('len(train_batch)): ' + str(len(train_batch)))
                                        
                    complete_combat_data = np.vstack([train_gene_data,gene_data_pre])
                    combat.fit(Y = complete_combat_data, b = train_batch, X = None, C = None)
                    
                    gene_data_pre = combat.transform(Y = np.vstack([gene_data_pre,complete_combat_data]), b = [1] * len(label_pre) + train_batch, X = None, C = None)
                    gene_data_pre = gene_data_pre[0:len(label_pre),:]
                    train_gene_data = combat.transform(Y = np.vstack([train_gene_data,complete_combat_data]), b = [0] * len(train_label) + train_batch, X = None, C = None)
                    train_gene_data = train_gene_data[0:len(train_label),:]
                    val_gene_data = combat.transform(Y = np.vstack([val_gene_data,complete_combat_data]), b = [0] * len(val_lab) + train_batch, X = None, C = None)
                    val_gene_data = val_gene_data[0:len(val_lab),:]
                    tune_gene_data = combat.transform(Y = np.vstack([tune_gene_data,complete_combat_data]), b = [0] * len(tune_label) + train_batch, X = None, C = None) 
                    tune_gene_data = tune_gene_data[0:len(tune_label),:]
                    
                # transform input
                drug_data_pre[drug_data_pre >= vocab_size] = 0            
                model.fit([drug_data_pre,np.zeros([gene_data_pre.shape[0],1]),gene_data_pre],label_pre,epochs = epochs_pretrain,
                                batch_size = batch_size,verbose = verbose)
            
            early_stopping = EarlyStopping(monitor='loss', patience=early_stopping_patience)
            model.fit([train_drug_data,np.zeros([train_drug_data.shape[0],1]),train_gene_data],train_label,epochs = epochs,
                                 validation_data=([tune_drug_data,np.zeros([tune_drug_data.shape[0],1]),tune_gene_data],tune_label),
                                 batch_size = batch_size,
                                 callbacks = [early_stopping],
                                 shuffle=True, verbose = verbose)

            predictions = model.predict([val_drug_data,np.zeros([val_drug_data.shape[0],1]),val_gene_data])
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators = num_trees,
                                          n_jobs = -1)
            model.fit(np.hstack([train_gene_data, train_drug_data]),train_label)
            predictions = model.predict(np.hstack([val_gene_data,val_drug_data]))
            
        gt_complete.append(val_label)
        pred_complete.append(predictions)
        inhib_data_complete.append(val_inhib)
        lab_data_complete.append(val_lab)
    
        counter += 1
    return (gt_complete, pred_complete, inhib_data_complete, lab_data_complete)
    
    

def get_cv_result_multiple_models(n_splits,train_data,pre_train_data=None,
                 model_params = dict(),epochs_pretrain = 10, epochs = 100,
                 cv_key = 'lab_data',
                 batch_size = 64,
                 percentage_tune = 0.1,
                 early_stopping_patience = 5,
                 verbose = 1, num_use_train = None,
                 use_combat = False, transform_gene_data = False,
                 model_types = ['nn_baseline','nn_paccmann','rf'],
                 only_return_splits = False):
                 
    result_dict = dict()
    if 'rf' in model_types:
        num_trees = model_params['num_trees']
        from sklearn.ensemble import RandomForestRegressor
    if 'nn_baseline' in model_types:
        model_type = 'nn'
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
    
    gene_data  = train_data['gene_data']
    if transform_gene_data:
        gene_data   = utils.transformation_np(gene_data)
    drug_data       = train_data['drug_data']
    drug_data_des   = train_data['drug_data_des']
    label           = train_data['label']
    inhib_data      = train_data['inhib_data']
    lab_data        = train_data['lab_data']
    
    # transform input
    drug_data[drug_data >= vocab_size] = 0
    
    cv_out = GroupKFold(n_splits=n_splits)
    if cv_key is not None:
        cv_out_splits = cv_out.split(gene_data, label, train_data[cv_key])
    else:
        cv_out_splits = cv_out.split(gene_data, label, np.arange(len(label)))
    counter = 1
    
    trains = []
    tests = []
    for train, test in tqdm(cv_out_splits,total = n_splits):
        # select only subset of train data
        # TODO: select according to sheme
        if num_use_train is not None and num_use_train > 0:
            rand_idx = np.random.permutation(np.arange(len(train)))
            train = np.array(train)[rand_idx[0:num_use_train]]            
            
        train_label         = label[train]
        train_gene_data     = gene_data[train,:]
        train_drug_data     = drug_data[train,:]
        train_drug_des_data = drug_data_des[train,:]
        
        if only_return_splits:
            trains.append(train)
            tests.append(test)
            continue
        
        
        # select tuning data
        np.random.seed(counter)
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
        
        val_label = label[test]
        val_gene_data = gene_data[test,:]
        val_drug_data = drug_data[test,:]
        val_drug_des_data = drug_data_des[test,:]
        val_inhib = inhib_data[test]
        val_lab = lab_data[test]
        test_batch = [0] * len(test)
        
        # perform combat
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
                #print('train_gene_data.shape: ' + str(train_gene_data.shape))
                #print('gene_data_pre.shape: ' + str(gene_data_pre.shape))
                #print('np.vstack([train_gene_data,gene_data_pre]).shape: ' + str(np.vstack([train_gene_data,gene_data_pre]).shape))
                #print('len(train_batch)): ' + str(len(train_batch)))
                                    
                complete_combat_data = np.vstack([train_gene_data,gene_data_pre])
                combat.fit(Y = complete_combat_data, b = train_batch, X = None, C = None)
                
                gene_data_pre = combat.transform(Y = np.vstack([gene_data_pre,complete_combat_data]), b = [1] * len(label_pre) + train_batch, X = None, C = None)
                gene_data_pre = gene_data_pre[0:len(label_pre),:]
                train_gene_data = combat.transform(Y = np.vstack([train_gene_data,complete_combat_data]), b = [0] * len(train_label) + train_batch, X = None, C = None)
                train_gene_data = train_gene_data[0:len(train_label),:]
                val_gene_data = combat.transform(Y = np.vstack([val_gene_data,complete_combat_data]), b = [0] * len(val_lab) + train_batch, X = None, C = None)
                val_gene_data = val_gene_data[0:len(val_lab),:]
                tune_gene_data = combat.transform(Y = np.vstack([tune_gene_data,complete_combat_data]), b = [0] * len(tune_label) + train_batch, X = None, C = None) 
                tune_gene_data = tune_gene_data[0:len(tune_label),:]
                # transform input
                drug_data_pre[drug_data_pre >= vocab_size] = 0     
        
        use_modes = ['scratch','pretrain']
        if num_use_train is not None and num_use_train == 0:
            use_modes = ['pretrain']
        
        
        # for all model_types
        for model_type in model_types:
            if type(epochs_pretrain) == dict:
                use_epochs_pretrain = epochs_pretrain[model_type]
            else:
                use_epochs_pretrain = epochs_pretrain
        
            if model_type == 'nn_baseline':
                for mode in use_modes:
                    tf.keras.backend.clear_session()
                    model = models.get_baseline_nn_model(num_gene_features = num_gene_features,
                        vocab_size = vocab_size,
                        drug_len = drug_len,
                        embed_dim = embed_dim,
                        drug_filters = drug_filters,
                        drug_kernels = drug_kernels,
                        dense_layers = dense_layers,
                        pool_sizes = pool_sizes,
                        use_batch_decorrelation = use_batch_decorrelation,
                        use_normal_batch_norm = use_normal_batch_norm)
                    
                    if mode == 'pretrain':                        
                        model.fit([gene_data_pre,drug_data_pre],label_pre,epochs = use_epochs_pretrain,
                                        batch_size = batch_size,verbose = verbose)
                       
                    if num_use_train is not None and num_use_train == 0:
                        predictions = model.predict([val_gene_data,val_drug_data])
                    else:
                        early_stopping = EarlyStopping(monitor='loss', patience=early_stopping_patience)
                        model.fit([train_gene_data,train_drug_data],train_label,epochs = epochs,
                                             validation_data=([tune_gene_data,tune_drug_data],tune_label),
                                             batch_size = batch_size,
                                             callbacks = [early_stopping],
                                             shuffle=True, verbose = verbose)
                        predictions = model.predict([val_gene_data,val_drug_data])
                    cur_model_name = model_type + '_' + mode
                    if cur_model_name not in result_dict:
                        result_dict[cur_model_name] = dict()
                    if 'gt_complete' not in result_dict[cur_model_name]:
                        result_dict[cur_model_name]['gt_complete'] = []
                        result_dict[cur_model_name]['pred_complete'] = []
                        result_dict[cur_model_name]['inhib_data_complete'] = []
                        result_dict[cur_model_name]['lab_data_complete'] = []
                    
                    result_dict[cur_model_name]['gt_complete'].append(val_label)
                    result_dict[cur_model_name]['pred_complete'].append(predictions)
                    result_dict[cur_model_name]['inhib_data_complete'].append(val_inhib)
                    result_dict[cur_model_name]['lab_data_complete'].append(val_lab)
                    
            elif model_type == 'nn_paccmann':
                for mode in use_modes:
                    tf.keras.backend.clear_session()
                    model = paccmann_model.get_paccmann_model(model_params)
                    
                    if mode == 'pretrain':                               
                        model.fit([drug_data_pre,np.zeros([gene_data_pre.shape[0],1]),gene_data_pre],label_pre,epochs = use_epochs_pretrain,
                                        batch_size = batch_size,verbose = verbose)
                        
                    if num_use_train is not None and num_use_train == 0:
                        predictions = model.predict([val_drug_data,np.zeros([val_drug_data.shape[0],1]),val_gene_data])
                    else:
                        early_stopping = EarlyStopping(monitor='loss', patience=early_stopping_patience)
                        model.fit([train_drug_data,np.zeros([train_drug_data.shape[0],1]),train_gene_data],train_label,epochs = epochs,
                                             validation_data=([tune_drug_data,np.zeros([tune_drug_data.shape[0],1]),tune_gene_data],tune_label),
                                             batch_size = batch_size,
                                             callbacks = [early_stopping],
                                             shuffle=True, verbose = verbose)

                        predictions = model.predict([val_drug_data,np.zeros([val_drug_data.shape[0],1]),val_gene_data])
                    
                    cur_model_name = model_type + '_' + mode
                    if cur_model_name not in result_dict:
                        result_dict[cur_model_name] = dict()
                    if 'gt_complete' not in result_dict[cur_model_name]:
                        result_dict[cur_model_name]['gt_complete'] = []
                        result_dict[cur_model_name]['pred_complete'] = []
                        result_dict[cur_model_name]['inhib_data_complete'] = []
                        result_dict[cur_model_name]['lab_data_complete'] = []
                    
                    result_dict[cur_model_name]['gt_complete'].append(val_label)
                    result_dict[cur_model_name]['pred_complete'].append(predictions)
                    result_dict[cur_model_name]['inhib_data_complete'].append(val_inhib)
                    result_dict[cur_model_name]['lab_data_complete'].append(val_lab)
            elif model_type == 'tDNN':
                for mode in use_modes:
                    tf.keras.backend.clear_session()
                    model = models.get_tdnn_model(num_gene_features = model_params['genes_number'],
                                            num_drug_features = model_params['drug_descriptors'])
                    
                    if mode == 'pretrain':                               
                        model.fit([drug_data_des_pre,gene_data_pre],label_pre,epochs = use_epochs_pretrain,
                                        batch_size = batch_size,verbose = verbose)
                        
                    if num_use_train is not None and num_use_train == 0:
                        predictions = model.predict([val_drug_des_data,val_gene_data])
                    else:
                        early_stopping = EarlyStopping(monitor='loss', patience=early_stopping_patience)
                        model.fit([train_drug_des_data,train_gene_data],train_label,epochs = epochs,
                                             validation_data=([tune_drug_des_data,tune_gene_data],tune_label),
                                             batch_size = batch_size,
                                             callbacks = [early_stopping],
                                             shuffle=True, verbose = verbose)

                        predictions = model.predict([val_drug_des_data,val_gene_data])
                    
                    cur_model_name = model_type + '_' + mode
                    if cur_model_name not in result_dict:
                        result_dict[cur_model_name] = dict()
                    if 'gt_complete' not in result_dict[cur_model_name]:
                        result_dict[cur_model_name]['gt_complete'] = []
                        result_dict[cur_model_name]['pred_complete'] = []
                        result_dict[cur_model_name]['inhib_data_complete'] = []
                        result_dict[cur_model_name]['lab_data_complete'] = []
                    
                    result_dict[cur_model_name]['gt_complete'].append(val_label)
                    result_dict[cur_model_name]['pred_complete'].append(predictions)
                    result_dict[cur_model_name]['inhib_data_complete'].append(val_inhib)
                    result_dict[cur_model_name]['lab_data_complete'].append(val_lab)
            elif model_type == 'rf':
                if num_use_train is not None and num_use_train == 0:
                    continue
                model = RandomForestRegressor(n_estimators = num_trees,
                                              n_jobs = -1)
                model.fit(np.hstack([train_gene_data, train_drug_data]),train_label)
                predictions = model.predict(np.hstack([val_gene_data,val_drug_data]))
                
                cur_model_name = model_type
                if cur_model_name not in result_dict:
                    result_dict[cur_model_name] = dict()
                if 'gt_complete' not in result_dict[cur_model_name]:
                    result_dict[cur_model_name]['gt_complete'] = []
                    result_dict[cur_model_name]['pred_complete'] = []
                    result_dict[cur_model_name]['inhib_data_complete'] = []
                    result_dict[cur_model_name]['lab_data_complete'] = []
                
                result_dict[cur_model_name]['gt_complete'].append(val_label)
                result_dict[cur_model_name]['pred_complete'].append(predictions)
                result_dict[cur_model_name]['inhib_data_complete'].append(val_inhib)
                result_dict[cur_model_name]['lab_data_complete'].append(val_lab)
    
        counter += 1
    if only_return_splits:
        return trains, tests
    return result_dict