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

def get_train_pretrain_data(param_json_path = 'data/params.json',
                            source='gdsc',
                            target='beat_aml',
                            use_netpop='ensemble',
                            seed = 42,
                            flag_normalize_descriptors='True',
                            train_mode='drug_repurposing',
                            root_dir = ''):
                                
    # read params
    with open(param_json_path) as json_file:
        param_dict = json.load(json_file)

        
    ## Load gene expression data for source and target
    # load gene expression data for source
    if source == 'gdsc':
        source_expression_dict = utils.load_expression_data(file_path = param_dict[source + '_rma_data_path'],
                                                     gene_col = 'GENE_SYMBOLS',
                                                     start_offset = 2,
                                                     sep = '\t',
                                                     root_dir = root_dir)
    else:
        print('source ' + source + ' not implemented yet!')
        
    if target == 'xenografts':
        target_expression_dict = utils.load_expression_data(file_path = param_dict[target + '_rna_data_path'],
                                                     gene_col = 'HGNC',
                                                     start_offset = 2,
                                                     sep = '\t',
                                                     root_dir = root_dir)
    elif target == 'gdsc':
        target_expression_dict = utils.load_expression_data(file_path = param_dict[target + '_rma_data_path'],
                                                     gene_col = 'GENE_SYMBOLS',
                                                     start_offset = 2,
                                                     sep = '\t',
                                                     root_dir = root_dir)
    elif target == 'ccle':
        target_expression_dict = utils.load_expression_data(file_path = param_dict[target + '_rna_data_path'],
                                                     gene_col = 'Description',
                                                     start_offset = 2,
                                                     sep = '\t',
                                                     root_dir = root_dir)
    elif target == 'pancreas':
        pancreas_gene_symbol_mapping_data = pd.read_csv(param_dict['pancreas_gene_symbol_mapping_path'],sep='\t')
        gene_symbol_mapping = dict()
        genes = list(pancreas_gene_symbol_mapping_data['gene'])
        symbols = list(pancreas_gene_symbol_mapping_data['symbol'])
        for i in range(len(genes)):
            if str(symbols[i]) != 'nan':
                gene_symbol_mapping[genes[i]] = symbols[i]
        
        
        target_expression_dict = utils.load_expression_data(file_path = param_dict[target + '_rna_data_path'],
                                                     gene_col = 'id_0',
                                                     start_offset = 2,
                                                     sep = '\t',
                                                     gene_mapping = gene_symbol_mapping,
                                                     root_dir = root_dir)
    elif target == 'beat_aml':
        target_expression_dict = utils.load_expression_data(file_path = param_dict['beat_rna_data_path'],
                                                     gene_col = 'Symbol',
                                                     start_offset = 2,
                                                     sep = ',',
                                                     root_dir = root_dir)
    else:
        print('target ' + target + ' not implemented yet!')
        return
    
    # get genes present in both dataset
    genes_shared = list(set(source_expression_dict['raw_symbols']).intersection(set(target_expression_dict['raw_symbols'])))
    print('number of genes in ' + source + ' and ' + target + ': ' + str(len(genes_shared)))
    gene_list = genes_shared
    
    # create dataframes with shared genes
    source_expression_df = utils.create_df_for_gene_list(source_expression_dict,genes_shared,verbose=1)
    target_expression_df = utils.create_df_for_gene_list(target_expression_dict,genes_shared,verbose=1)
    
    
    # create dictionary containing the expression features for cell lines in source and target
    cellline_vector_dict = dict()
    if source == 'gdsc' or target == 'gdsc':
        if source == 'gdsc':
            cur_data = source_expression_df
        else:
            cur_data = target_expression_df
        values = cur_data.values
        tmp_celllines = list(cur_data.columns)
        for i in tqdm(np.arange(len(tmp_celllines))):
            cellline_vector_dict[tmp_celllines[i].replace('DATA.','')] = values[:,i]


    if source == 'xenografts' or  target == 'xenografts':
        if source == 'xenografts':
            cur_data = source_expression_df
        else:
            cur_data = target_expression_df 
        values = cur_data.values
        tmp_celllines = list(cur_data.columns)
        for i in tqdm(np.arange(len(tmp_celllines))):
            cur_line = tmp_celllines[i]
            cur_pat = cur_line.split('.')[0]
            if cur_line.startswith(str(cur_pat) + '.X') and cur_line.endswith('intensity'):
                cellline_vector_dict[cur_pat] = values[:,i]
                
    if source == 'ccle' or  target == 'ccle':
        if source == 'ccle':
            cur_data = source_expression_df
        else:
            cur_data = target_expression_df 
        values = cur_data.values
        tmp_celllines = list(cur_data.columns)
        for i in tqdm(np.arange(len(tmp_celllines))):
            cur_pat = tmp_celllines[i]
            cellline_vector_dict[cur_pat] = values[:,i]
            
    if source == 'pancreas' or  target == 'pancreas':
        pancreas_filename_organoid_data = pd.read_csv(root_dir + param_dict['pancreas_filename_organoid_path'],sep='\t')
        pancreas_filename_organoid_data.head()
        filename_organoid_mapping = dict()
        filenames = list(pancreas_filename_organoid_data['filename'])
        organoid = list(pancreas_filename_organoid_data['organoid'])
        for i in range(len(filenames)):
            filename_organoid_mapping[filenames[i]] = organoid[i]
        
        
        if source == 'pancreas':
            cur_data = source_expression_df
        else:
            cur_data = target_expression_df 
        values = cur_data.values
        tmp_celllines = list(cur_data.columns)
        for i in tqdm(np.arange(len(tmp_celllines))):
            cellline_vector_dict[filename_organoid_mapping[tmp_celllines[i] + '.gz']] = values[:,i]
            
            
    if source == 'beat_aml' or  target == 'beat_aml':
        if source == 'beat_aml':
            cur_data = source_expression_df
        else:
            cur_data = target_expression_df 
        values = cur_data.values
        tmp_celllines = list(cur_data.columns)
        for i in tqdm(np.arange(len(tmp_celllines))):
            cellline_vector_dict[tmp_celllines[i]] = values[:,i]
    
    
    ## Collect features for drugs
    drug_smiles_dict = dict()
    if source == 'gdsc' or target == 'gdsc':
        source_inchi_data = pd.read_csv(root_dir + param_dict['gdsc_inchi_path'])
        tmp_drug_names = list(source_inchi_data['inhibitor'])
        tmp_smiles = list(source_inchi_data['canonical_smiles'])
        for i in tqdm(np.arange(len(tmp_drug_names))):
            if str(tmp_smiles[i]) == 'nan':
                continue
            drug_smiles_dict[tmp_drug_names[i]] = tmp_smiles[i]        
            
    if source == 'xenografts' or target == 'xenografts':
        target_inchi_data = pd.read_csv(root_dir + param_dict['xenografts_inchi_path'])
        tmp_drug_names = list(target_inchi_data['inhibitor'])
        tmp_smiles = list(target_inchi_data['canonical_smiles'])
        for i in tqdm(np.arange(len(tmp_drug_names))):
            if str(tmp_smiles[i]) == 'nan':
                continue
            drug_smiles_dict[tmp_drug_names[i]] = tmp_smiles[i]

    if source == 'ccle' or target == 'ccle':
        target_inchi_data = pd.read_csv(root_dir + param_dict['ccle_inchi_path'])
        tmp_drug_names = list(target_inchi_data['inhibitor'])
        tmp_smiles = list(target_inchi_data['canonical_smiles'])
        for i in tqdm(np.arange(len(tmp_drug_names))):
            if str(tmp_smiles[i]) == 'nan':
                continue
            drug_smiles_dict[tmp_drug_names[i]] = tmp_smiles[i]
            
    if source == 'pancreas' or target == 'pancreas':
        target_inchi_data = pd.read_csv(root_dir + param_dict['pancreas_inchi_path'])
        tmp_drug_names = list(target_inchi_data['inhibitor'])
        tmp_smiles = list(target_inchi_data['canonical_smiles'])
        for i in tqdm(np.arange(len(tmp_drug_names))):
            if str(tmp_smiles[i]) == 'nan':
                continue
            drug_smiles_dict[tmp_drug_names[i]] = tmp_smiles[i]

            
    if source == 'beat_aml' or target == 'beat_aml':
        target_inchi_data = pd.read_csv(root_dir + param_dict['beat_inchi_path'])
        tmp_drug_names = list(target_inchi_data['inhibitor'])
        tmp_smiles = list(target_inchi_data['canonical_smiles'])
        for i in tqdm(np.arange(len(tmp_drug_names))):
            if str(tmp_smiles[i]) == 'nan':
                continue
            drug_smiles_dict[tmp_drug_names[i]] = tmp_smiles[i]        

    print('number of drugs: ' + str(len(drug_smiles_dict.keys())))
    
    
    ## Collect molecure descriptors
    molecule_descriptor_df = pd.read_csv(root_dir + param_dict['molecule_descriptor_path'],sep='\t')
    drug_descriptor_dict = dict()
    inhib_list = list(molecule_descriptor_df['inhibitor'])
    feature_matrix = np.array(molecule_descriptor_df.values[:,2:],dtype=np.float32)
    feature_matrix[np.isnan(feature_matrix)] = 0

    min_max_scaler_descr = MinMaxScaler()
    if flag_normalize_descriptors:        
        feature_matrix = min_max_scaler_descr.fit_transform(feature_matrix)

    for i in range(len(inhib_list)):
        drug_descriptor_dict[inhib_list[i]] = feature_matrix[i,:]
        
    num_descr_features = feature_matrix.shape[1]
    
    ## Create feature representation for smiles and store it in a dictionary
    smiles_character_dict = dict()
    smile_lens = [len(drug_smiles_dict[drug]) for drug in drug_smiles_dict]
    for drug in drug_smiles_dict:
        cur_smiles = drug_smiles_dict[drug]
        for char in cur_smiles:
            if char not in smiles_character_dict:
                smiles_character_dict[char] = len(smiles_character_dict) + 1

    max_smiles_len = np.max(smile_lens)

    drug_smiles_vec_dict = dict()
    for drug in drug_smiles_dict:
        cur_smiles = drug_smiles_dict[drug]
        cur_vec = np.zeros([max_smiles_len,])
        for i in range(np.min([len(cur_smiles),max_smiles_len])):
            char = cur_smiles[i]
            if char in smiles_character_dict:
                cur_val = smiles_character_dict[char]
            else:
                cur_val = 0
            cur_vec[i] = cur_val
        drug_smiles_vec_dict[drug] = cur_vec

    
    
    ## Collect data for source and target
    # collect data for target
    if target == 'xenografts':
        xenografts_data = pd.read_csv(root_dir + param_dict['xenografts_data_path'],sep='\t')
        lab_ids = list(xenografts_data['pat_id'])
        inhibitors = list(xenografts_data['inhibitor'])
        values = np.array(list(xenografts_data['value']),dtype = np.float32)
    elif target == 'ccle':
        ccle_data = pd.read_csv(root_dir + param_dict['ccle_data_path'],sep='\t')
        lab_ids = list(ccle_data['pat_id'])
        inhibitors = list(ccle_data['inhibitor'])
        values = np.array(list(ccle_data['value']),dtype = np.float32)
    elif target == 'pancreas':
        pancreas_data = pd.read_csv(root_dir + param_dict['pancreas_data_path'],sep='\t')
        lab_ids = list(pancreas_data['organoid'])
        inhibitors = list(pancreas_data['inhibitor'])
        values = np.array(list(pancreas_data['value']),dtype = np.float32)
    elif target == 'beat_aml':
        beat_data = pd.read_csv(root_dir + param_dict['beat_data_path'])
        lab_ids = list(beat_data['lab_id'])
        inhibitors = list(beat_data['inhibitor'])
        values = np.array(list(beat_data['auc']),dtype = np.float32)
    elif target == 'gdsc':
        gdsc_data = pd.read_excel(param_dict['gdsc_data_path'])
        lab_ids = list(gdsc_data['COSMIC_ID'])
        inhibitors = list(gdsc_data['inhibitor'])
        values = np.array(list(gdsc_data['AUC']),dtype = np.float32)
    
    target_inhibitors = inhibitors
    target_labs_ids = lab_ids
    min_max_scaler = MinMaxScaler()
    values_min_max = min_max_scaler.fit_transform(values.reshape(-1, 1))
    values_min_max = np.reshape(values_min_max,[values_min_max.shape[0],])

    gene_data     = np.zeros([len(lab_ids),len(gene_list)])
    drug_data     = np.zeros([len(lab_ids),max_smiles_len])
    drug_data_des = np.zeros([len(lab_ids),num_descr_features])
    label         = np.zeros([len(lab_ids),])
    label_raw     = np.zeros([len(lab_ids),])
    inhib_data    = []
    lab_data      = []    
    counter       = 0
    for i in range(len(lab_ids)):
        try:
            cur_lab = str(int(lab_ids[i]))
        except:
            cur_lab = str(lab_ids[i])
        cur_inh = inhibitors[i]
        cur_value = values_min_max[i]
        if cur_lab in cellline_vector_dict and cur_inh in drug_smiles_vec_dict:
            gene_data[counter,:] = cellline_vector_dict[cur_lab]
            drug_data[counter,:] = drug_smiles_vec_dict[cur_inh]
            drug_data_des[counter,:] = drug_descriptor_dict[cur_inh]
            label[counter]       = cur_value
            label_raw[counter]   = values[i]
            inhib_data.append(cur_inh)
            lab_data.append(cur_lab)
            counter += 1
    label = label[0:counter]
    label_raw = label_raw[0:counter]
    gene_data = gene_data[0:counter]
    drug_data = drug_data[0:counter]
    drug_data_des = drug_data_des[0:counter]
    inhib_data = np.array(inhib_data)
    lab_data = np.array(lab_data,dtype=np.str)    
    
    print('number of cell lines/patients: ' + str(len(np.unique([str(gene_data[i,:]) for i in range(gene_data.shape[0])]))))
    print('number of drugs: ' + str(len(np.unique([str(drug_data[i,:]) for i in range(drug_data.shape[0])]))))
    print('number of samples: ' + str(gene_data.shape[0]))
    
    
    # collect data for source
    if source == 'gdsc':
        gdsc_data = pd.read_excel(root_dir + param_dict['gdsc_data_path'])
        lab_ids = list(gdsc_data['COSMIC_ID'])
        inhibitors = list(gdsc_data['DRUG_NAME'])
        aucs = np.array(list(gdsc_data['AUC']),dtype = np.float32)
        
        
        if target == 'ccle' and train_mode == 'precision_oncology':
            ccle_cell_line_names =  []
            for i in range(len(lab_data)):
                ccle_cell_line_names.append(lab_data[i].split('_')[0].strip())
            ccle_cell_line_set = set(ccle_cell_line_names)

            gdsc_cell_line_names = list(gdsc_data['CELL_LINE_NAME'])
            use_ids = []
            exclude_names = []
            for i in range(len(gdsc_cell_line_names)):
                if gdsc_cell_line_names[i] in ccle_cell_line_set:
                    exclude_names.append(gdsc_cell_line_names[i])
                    continue
                use_ids.append(i)
            exclude_names = set(exclude_names)
            print('number of gdsc cell-lines excluded: ' + str(len(exclude_names)))
            print('number of training examples used: ' + str(len(use_ids)) + ' [' +\
                              str(len(use_ids) / len(gdsc_cell_line_names)) + '%]')
            
            lab_ids = list(np.array(lab_ids)[use_ids])
            inhibitors = list(np.array(inhibitors)[use_ids])
            aucs = np.array(aucs)[use_ids]
            
        if train_mode == 'drug_development':
            target_inhibitor_list = list(np.unique(target_inhibitors))
            target_smiles = set()
            for target_inhibitor in target_inhibitor_list:
                try:
                    target_smiles.add(drug_smiles_dict[target_inhibitor])
                except:
                    continue
            #print('number of target smiles collected: ' + str(len(target_smiles)))
            
            use_ids = []
            exclude_names = []
            for i in range(len(inhibitors)):
                try:
                    cur_smile = drug_smiles_dict[inhibitors[i]]
                except:
                    cur_smile = np.nan
                if cur_smile in target_smiles:        
                    exclude_names.append(inhibitors[i])
                    continue
                use_ids.append(i)
            exclude_names = set(exclude_names)
            print('number of gdsc drugs excluded: ' + str(len(exclude_names)))
            print('number of training examples used: ' + str(len(use_ids)) + ' [' +\
                              str(len(use_ids) / len(inhibitors)) + '%]')
            
            lab_ids = list(np.array(lab_ids)[use_ids])
            inhibitors = list(np.array(inhibitors)[use_ids])
            aucs = np.array(aucs)[use_ids]

    gene_data_source     = np.zeros([len(lab_ids),len(gene_list)])
    drug_data_source     = np.zeros([len(lab_ids),max_smiles_len])
    drug_data_des_source = np.zeros([len(lab_ids),num_descr_features])
    label_source         = np.zeros([len(lab_ids),])
    inhib_data_source    = []
    lab_data_source      = []
    counter    = 0
    for i in range(len(lab_ids)):
        cur_lab = str(lab_ids[i])
        cur_inh = inhibitors[i]
        cur_auc = aucs[i]
        if cur_lab in cellline_vector_dict and cur_inh in drug_smiles_vec_dict:
            gene_data_source[counter,:] = cellline_vector_dict[cur_lab]
            drug_data_source[counter,:] = drug_smiles_vec_dict[cur_inh]
            drug_data_des_source[counter,:] = drug_descriptor_dict[cur_inh]
            label_source[counter]       = cur_auc
            inhib_data_source.append(cur_inh)
            lab_data_source.append(cur_lab)
            counter += 1
    label_source = label_source[0:counter]
    gene_data_source = gene_data_source[0:counter]
    drug_data_source = drug_data_source[0:counter]
    drug_data_des_source = drug_data_des_source[0:counter]
    inhib_data_source = np.array(inhib_data_source)
    lab_data_source = np.array(lab_data_source)
    
    # randomize data
    np.random.seed(seed) 
    rand_ids = np.arange(gene_data.shape[0])
    np.random.shuffle(rand_ids) 

    label         = label[rand_ids]
    label_raw     = label_raw[rand_ids]
    gene_data     = gene_data[rand_ids,:]
    drug_data     = drug_data[rand_ids,:]
    drug_data_des = drug_data_des[rand_ids,:]
    inhib_data    = inhib_data[rand_ids]
    lab_data      = lab_data[rand_ids]
    
    ## Load gene lists
    gene_list_dict = {'all':gene_list}
    # add paccmann
    paccmann_gene_list = list(pd.read_csv(root_dir + param_dict['paccmann_gene_list'],header=None)[0])
    gene_list_dict['paccmann'] = paccmann_gene_list

    network_prop_keys = ['netcore_sig_literature_mining',
                         'netcore_sig_gdsc_drug_targets_literature_mining',
                         'netcore_sig_gdsc_drug_targets']
    num_genes_per_drug_list = [10,20,30]

    # add network propagation gene lists
    for netcore_key in network_prop_keys:
        for num_genes_per_drug in num_genes_per_drug_list:
            genes_use = utils.get_gene_list_for_network_prop_df(root_dir + param_dict[netcore_key],
            num_genes_per_drug = num_genes_per_drug,
            min_weight_gene = None)
            gene_list_dict[netcore_key + '_' + str(num_genes_per_drug)] = genes_use
            
    # add genes from ensemble learning
    gdsc_genes = list(np.array(pd.read_csv(root_dir + param_dict['gdsc_gene_list'],header=None).values[:,0]))
    ocokb_genes = list(np.array(pd.read_csv(root_dir + param_dict['oncokb_gene_list'],sep='\t').values[:,0]))
    lincs_genes = list(np.array(pd.read_csv(root_dir + param_dict['lincs_gene_list'],header=None).values[:,0]))
    gene_list_dict['ensemble'] = list(set(gdsc_genes + ocokb_genes + lincs_genes))
    print('len(gene_list_ensemble): ' + str(len(gene_list_dict['ensemble'])))
    
    # get model_params
    result_path = root_dir + param_dict['model_param_pretrain_csv']

    #use_netpop = None
    model_params, gene_key, gene_use_ids = models.get_best_model_params(result_path,
                         gene_list_dict = gene_list_dict,
                         complete_gene_list = gene_list,
                         gene_list = use_netpop)
    
    # create train test data
    train_data = {'gene_data': gene_data[:,gene_use_ids],
                  'drug_data': drug_data,
                  'drug_data_des':drug_data_des,
                  'label': label,
                  'inhib_data': inhib_data,
                  'lab_data':lab_data,
                  'label_raw':label_raw}


    pre_train_data = {'gene_data': gene_data_source[:,gene_use_ids],
                  'drug_data': drug_data_source,
                  'drug_data_des':drug_data_des_source,
                  'label': label_source,
                  'inhib_data': inhib_data_source,
                  'lab_data':lab_data_source}
    
    return {'train_data':train_data,
            'pre_train_data':pre_train_data,
            'min_max_scaler': min_max_scaler,
            'min_max_scaler_descr' :  min_max_scaler_descr,
            'gene_list_dict':gene_list_dict,
            'gene_list':gene_list,
            'smiles_character_dict':smiles_character_dict,
            'gene_list_used':list(np.array(gene_list)[gene_use_ids]),
            }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-GPU','--GPU',type=int,default=0)    
    parser.add_argument('-param_json_path', '--param_json_path', type=str, default='data/params.json')    
    parser.add_argument('-source', '--source', type=str, default='gdsc')    
    parser.add_argument('-target', '--target', type=str, default='beat_aml')    
    parser.add_argument('-use_netpop', '--use_netpop', type=str, default='ensemble')
    parser.add_argument('-model_types', '--model_types', type=str, default="['tDNN','nn_baseline','nn_paccmann']")    # "['tDNN','nn_baseline','nn_paccmann','rf']"
    parser.add_argument('-seed','--seed', type=int, default = 42)
    parser.add_argument('-n_splits','--n_splits', type=int, default = 10)    
    parser.add_argument('-flag_normalize_descriptors','--flag_normalize_descriptors',type=str,default='True')
    parser.add_argument('-use_samples','--use_samples',type=int,default=10000)
    parser.add_argument('-train_mode','--train_mode',type=str,default='drug_repurposing')
    parser.add_argument('-save_dir','--save_dir',type=str,default='results/')
    parser.add_argument('-save_prefix','--save_prefix',type=str,default='')
    parser.add_argument('-flag_redo','--flag_redo',type=str,default='True')
    parser.add_argument('-batch_size','--batch_size',type=int,default=256)
    
    
    args = parser.parse_args()
    GPU = args.GPU
    param_json_path = args.param_json_path
    source = args.source
    target = args.target
    use_netpop = args.use_netpop
    model_types = list_string(args.model_types)
    seed = args.seed
    n_splits = args.n_splits
    flag_normalize_descriptors = boolean_string(args.flag_normalize_descriptors)
    use_samples = args.use_samples
    train_mode = args.train_mode
    save_dir = args.save_dir
    save_prefix = args.save_prefix
    flag_redo = boolean_string(args.flag_redo)
    batch_size = args.batch_size
    
    save_path = save_dir +  save_prefix + source + '_' + target + '_' + str(use_samples) +\
                '_' + str(flag_normalize_descriptors) + '_' + str(train_mode) + '_' + str(use_netpop) + '.joblib'
                
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
    
    
    cur_train_pretrain_data_dict = get_train_pretrain_data(param_json_path = param_json_path,
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
    gene_list_used = cur_train_pretrain_data_dict['gene_list_used']
    
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

    epochs_pretrain = {'nn_baseline':100,
                       'nn_paccmann':100,
                       'rf':None,
                       'tDNN':100}

    # tDNN
    model_params.update({'drug_descriptors':train_data['drug_data_des'].shape[1]})
        
    early_stopping_patience = 25
    tmp_result_dict =  evaluation.get_cv_result_multiple_models(n_splits = n_splits,
                 train_data = train_data,pre_train_data=pre_train_data,
                 model_params = model_params,epochs_pretrain = epochs_pretrain, epochs = 1000,
                 cv_key = cv_key, batch_size = batch_size, num_use_train = use_samples,
                 use_combat = True, transform_gene_data = True,
                 model_types = model_types,
                 early_stopping_patience = 25)
    tmp_result_dict['min_max_scaler']       = min_max_scaler
    tmp_result_dict['min_max_scaler_descr'] = min_max_scaler_descr
    
    joblib.dump(tmp_result_dict, save_path, compress=3, protocol=2)
    
if __name__ == "__main__":
    # execute only if run as a script
    main() 