import tensorflow as tf
# layer for CNN
from tensorflow.keras.layers import Embedding, Attention, Concatenate, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Input, Dropout, Dense, Average, BatchNormalization, Flatten
# layer for LSTM
from tensorflow.keras.layers import LSTM, Bidirectional
# layer for Transformer
from tensorflow.keras import layers
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras import metrics
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np



def get_random_model_params(gene_list_dict = dict(),
                     complete_gene_list = [],
                     vocab_size = 28,
                     drug_len = 188,
                     embed_dim_list = [16,32,64,128],
                     conv_len = [1,2,3],
                     filter_sizes = [16,32,64],
                     kernel_sizes = [3,5,7,9,11],
                     pool_sizes = [2,4],
                     num_dense = [1,2,3],
                     dense_sizes = [32,64,128,256,512],
                     use_batch_decorrelation_list = [False],
                     use_normal_batch_norm_list = [True,False]):
    
    # get gene-list
    gene_keys = list(gene_list_dict.keys())
    gene_key = gene_keys[np.random.randint(len(gene_keys))]
    use_genes = gene_list_dict[gene_key]
    use_genes_set = set(use_genes)
    
    # get ids of genes to use
    gene_use_ids = []
    for i in range(len(complete_gene_list)):
        if complete_gene_list[i] in use_genes_set:
            gene_use_ids.append(i)
    
    # get embedding dim
    embed_dim = embed_dim_list[np.random.randint(len(embed_dim_list))]
    
    # get num of conv layers
    num_conv_layer = conv_len[np.random.randint(len(conv_len))]
    
    # get filter sizes, kernel sizes, and pool sizes
    #print('try to find filter sizes [' + str(num_conv_layer) + ']')
    while True:
        tmp_list = [kernel_sizes[np.random.randint(len(kernel_sizes))] for a in range(num_conv_layer)]
        tmp_sort_ids = np.argsort(tmp_list)
        ordering = np.all(np.array_equal(np.array(tmp_sort_ids), np.array([num_conv_layer - (a+1) for a in range(num_conv_layer)])))
        if ordering:
            break
    #print('done')
            
    filter_size  = filter_sizes[np.random.randint(len(filter_sizes))]
    pool_size    = pool_sizes[np.random.randint(len(pool_sizes))]
    drug_filters = [filter_size for a in range(num_conv_layer)]
    pool_sizes   = [pool_size for a in range(num_conv_layer)]
    drug_kernels = tmp_list
    
    
    # get dense layers
    num_dense_layer = num_dense[np.random.randint(len(num_dense))]
    
    #print('try to find dense sizes [' + str(num_dense_layer) + ']')
    while True:
        tmp_list = [dense_sizes[np.random.randint(len(dense_sizes))] for a in range(num_dense_layer)]
        tmp_sort_ids = np.argsort(tmp_list)
        ordering = np.all(np.array_equal(np.array(tmp_sort_ids),np.array([num_dense_layer - (a+1) for a in range(num_dense_layer)])))
        if ordering:
            break
    #print('done')
            
    dense_layers = tmp_list
    
    use_batch_decorrelation = use_batch_decorrelation_list[np.random.randint(len(use_batch_decorrelation_list))]
    use_normal_batch_norm = use_normal_batch_norm_list[np.random.randint(len(use_normal_batch_norm_list))]
    
    model_params = {'num_gene_features' : len(gene_use_ids),
                'vocab_size' : vocab_size,
                'drug_len' :drug_len,
                'embed_dim' :embed_dim,
                'drug_filters' : drug_filters,
                'drug_kernels' : drug_kernels,
                'pool_sizes' : pool_sizes,
                'dense_layers' :dense_layers,
                'use_batch_decorrelation': use_batch_decorrelation,
                'use_normal_batch_norm': use_normal_batch_norm,
                'gene_list':gene_key}
    
    return model_params, gene_key, gene_use_ids
    
    
def get_best_model_params(result_path,
                     gene_list_dict = dict(),
                     complete_gene_list = [],
                     gene_list = None):
    
    result_df = pd.read_csv(result_path)
    if gene_list is not None and gene_list != 'ensemble':
        result_df = result_df[result_df['gene_list'] == gene_list]
    result_df = result_df.sort_values('MSE_inhib')
    best_config = result_df.iloc[0]
    
    
    
    # get gene-list
    gene_key = best_config['gene_list']
    use_genes = gene_list_dict[gene_key]
    use_genes_set = set(use_genes)
    
    # get ids of genes to use
    gene_use_ids = []
    for i in range(len(complete_gene_list)):
        if complete_gene_list[i] in use_genes_set:
            gene_use_ids.append(i)
    
    # get embedding dim
    embed_dim = int(best_config['embed_dim'])
    
    # get conv layers
                
    drug_filters = list(np.array(best_config['drug_filters'].replace('[','').replace(']','').split(','),dtype=np.int32))
    pool_sizes   = list(np.array(best_config['pool_sizes'].replace('[','').replace(']','').split(','),dtype=np.int32))
    drug_kernels = list(np.array(best_config['drug_kernels'].replace('[','').replace(']','').split(','),dtype=np.int32))
    
    
    # get dense layers
    dense_layers = list(np.array(best_config['dense_layers'].replace('[','').replace(']','').split(','),dtype=np.int32))
    
    use_batch_decorrelation = bool(best_config['use_batch_decorrelation'])
    use_normal_batch_norm = bool(best_config['use_normal_batch_norm'])
    
    
    vocab_size = int(best_config['vocab_size'])
    drug_len = int(best_config['drug_len'])
    
    model_params = {'num_gene_features' : len(gene_use_ids),
                'vocab_size' : vocab_size,
                'drug_len' :drug_len,
                'embed_dim' :embed_dim,
                'drug_filters' : drug_filters,
                'drug_kernels' : drug_kernels,
                'pool_sizes' : pool_sizes,
                'dense_layers' :dense_layers,
                'use_batch_decorrelation': use_batch_decorrelation,
                'use_normal_batch_norm': use_normal_batch_norm,
                'gene_list':gene_key}
    
    return model_params, gene_key, gene_use_ids

def get_baseline_nn_model(num_gene_features,
    vocab_size,
    drug_len,
    # params
    embed_dim = 32,
    drug_filters = [64,64,64],
    drug_kernels = [3,5,7],
    dense_layers = [128,64],
    pool_sizes   = [4,4,4],
    use_batch_decorrelation = False,
    use_normal_batch_norm = False,
    flag_embedding_as_input = False):
    input_gene = layers.Input(shape=(num_gene_features), name = 'gene_input')
    gene_embedding = Dense(embed_dim, name = 'dense_gene_embedding')(input_gene)
    
    if flag_embedding_as_input:
        input_drug = layers.Input(shape=(drug_len,embed_dim), name = 'drug_input')
        drug_embedding = input_drug
    else:
        input_drug = layers.Input(shape=(drug_len,), name = 'drug_input')
        drug_embedding = Embedding(input_dim = vocab_size, output_dim=embed_dim,input_length = drug_len, name='drug_embedding')(input_drug)
    
    conv_list = []
    num_conv = len(drug_kernels)
    for i in range(num_conv):
        if i == 0:
            #print('drug_filters[i]: ' + str(drug_filters[i]))
            #print('drug_kernels[i]: ' + str(drug_filters[i]))
            drug_conv = Conv1D(filters=int(drug_filters[i]),
                                kernel_size=int(drug_kernels[i]),
                                padding='same', name = 'conv_' + str(i+1))(drug_embedding)        
            # max pooling
            drug_conv = MaxPooling1D(pool_size = int(pool_sizes[i]), name='max_pool_' + str(i+1))(drug_conv)
        else:
            drug_conv = Conv1D(filters=int(drug_filters[i]),
                                kernel_size=int(drug_kernels[i]),
                                padding='same', name = 'conv_' + str(i+1))(drug_conv)        
            # max pooling
            drug_conv = MaxPooling1D(pool_size = int(pool_sizes[i]), name='max_pool_' + str(i+1))(drug_conv)
    flatten_max = Flatten(name='pool_flatten_' + str(i+1))(drug_conv)
    conv_list.append(flatten_max)
    concat_list = conv_list + [gene_embedding]

    x = Concatenate(name = 'concat')(concat_list)

    for i in range(len(dense_layers)):
        cur_dense = int(dense_layers[i])
        x = layers.Dropout(0.1, name='dropout_' + str(i+1))(x)
        x = layers.Dense(cur_dense, activation="relu",name='concat_dense_' + str(i+1))(x)
        if use_normal_batch_norm:
            x = BatchNormalization(name='batch_normalization_' + str(i+1))(x)
        if use_batch_decorrelation:
            x = DecorelationNormalization(name='decorrelation_normalization_' + str(i+1))(x)
    x = layers.Dropout(0.1, name='dropout_pre_final')(x)
    outputs = Dense(1, activation='linear',name='final_dense')(x)
    
    model = tf.keras.Model(inputs=[input_gene, input_drug], outputs=outputs)
    
    opt = tf.keras.optimizers.Adam()#learning_rate=learning_rate)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer = opt, loss = loss, 
                metrics=["mse"])
    return model
    

def get_tdnn_model(num_gene_features,
                    num_drug_features):
    input_gene = layers.Input(shape=(num_gene_features), name = 'gene_input')
    input_drug = layers.Input(shape=(num_drug_features), name = 'drug_input')
    
    # gene
    gene_dense = layers.Dense(1000, activation="relu",name='first_dense_gene')(input_gene)
    gene_dense = layers.Dense(500, activation="relu",name='second_dense_gene')(gene_dense)
    gene_dense = layers.Dense(250, activation="relu",name='third_dense_gene')(gene_dense)
    
    # drugs
    drug_dense = layers.Dense(1000, activation="relu",name='first_dense_drug')(input_drug)
    drug_dense = layers.Dense(500, activation="relu",name='second_dense_drug')(drug_dense)
    drug_dense = layers.Dense(250, activation="relu",name='third_dense_drug')(drug_dense)
    
    # concatenate
    concat = Concatenate(name = 'concat')([gene_dense,drug_dense])
    
    concat_dense = layers.Dense(250, activation="relu",name='first_dense_concat_dense')(concat)
    concat_dense = layers.Dense(125, activation="relu",name='second_dense_concat_dense')(concat_dense)
    concat_dense = layers.Dense(60, activation="relu",name='third_dense_concat_dense')(concat_dense)
    outputs = layers.Dense(1, activation='linear',name='final_dense')(concat_dense)

    model = tf.keras.Model(inputs=[input_drug, input_gene], outputs=outputs)
    
    opt = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer = opt, loss = loss, 
                metrics=["mse"])
    return model