import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compute_gradients_model(model,data_points, target_class_idx):
    with tf.GradientTape() as tape:
        data_points_tensor_1 = tf.convert_to_tensor(data_points[0],dtype=tf.float32)
        data_points_tensor_2 = tf.convert_to_tensor(data_points[1],dtype=tf.float32)
        tape.watch(data_points_tensor_1)        
        probs = model([data_points_tensor_1,data_points_tensor_2])[:,target_class_idx]
    grad_1 = tape.gradient(probs, data_points_tensor_1).numpy()
    with tf.GradientTape() as tape:
        data_points_tensor_1 = tf.convert_to_tensor(data_points[0],dtype=tf.float32)
        data_points_tensor_2 = tf.convert_to_tensor(data_points[1],dtype=tf.float32)
        tape.watch(data_points_tensor_2)
        probs = model([data_points_tensor_1,data_points_tensor_2])[:,target_class_idx]
    grad_2 = tape.gradient(probs, data_points_tensor_2).numpy()
    return (grad_1, grad_2)
    
def compute_gradients_model_paccmann(model,data_points, target_class_idx):
    with tf.GradientTape() as tape:
        data_points_tensor_1 = tf.convert_to_tensor(data_points[0],dtype=tf.float32)
        data_points_tensor_2 = tf.convert_to_tensor(data_points[1],dtype=tf.float32)
        data_points_tensor_3 = tf.convert_to_tensor(data_points[2],dtype=tf.float32)
        tape.watch(data_points_tensor_1)        
        probs = model([data_points_tensor_1,data_points_tensor_2,data_points_tensor_3])[:,target_class_idx]
    grad_1 = tape.gradient(probs, data_points_tensor_1).numpy()
    '''
    with tf.GradientTape() as tape:
        data_points_tensor_1 = tf.convert_to_tensor(data_points[0],dtype=tf.float32)
        data_points_tensor_2 = tf.convert_to_tensor(data_points[1],dtype=tf.float32)
        data_points_tensor_3 = tf.convert_to_tensor(data_points[2],dtype=tf.float32)
        tape.watch(data_points_tensor_2)
        probs = model([data_points_tensor_1,data_points_tensor_2,data_points_tensor_3])[:,target_class_idx]
    grad_2 = tape.gradient(probs, data_points_tensor_2).numpy()
    '''
    with tf.GradientTape() as tape:
        data_points_tensor_1 = tf.convert_to_tensor(data_points[0],dtype=tf.float32)
        data_points_tensor_2 = tf.convert_to_tensor(data_points[1],dtype=tf.float32)
        data_points_tensor_3 = tf.convert_to_tensor(data_points[2],dtype=tf.float32)
        tape.watch(data_points_tensor_3)
        probs = model([data_points_tensor_1,data_points_tensor_2,data_points_tensor_3])[:,target_class_idx]
    grad_3 = tape.gradient(probs, data_points_tensor_3).numpy()
    return (grad_1, grad_3)
    
def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients
    
    
def interpolate_embeddings(example,
                           alphas,
                           embedding_size=128,
                           baseline_method='zeros'):
    if baseline_method == 'zeros':
        baseline = tf.identity(example).numpy()
        baseline[:,0:embedding_size] = 0 
    else:
        print('not implemented')
        return None
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    baseline_x = baseline
    input_x = example
    delta = input_x - baseline_x
    embeddings = baseline_x +  alphas_x * delta
    if embedding_size == 1:
        # delete second dimension
        embeddings = tf.reshape(embeddings, (embeddings.shape[0], embeddings.shape[2]))
    return embeddings
    
    
def get_attributions(orig_data, model_con, model_emb,alphas,
                     baseline_method='zeros'):
    # get embedding size for drug tokens
    embedding_size = model_con.outputs[0].shape[2]
    example = model_con(orig_data)
    drug_embeddings = interpolate_embeddings(example.numpy(),alphas,
                                              embedding_size=embedding_size,
                                              baseline_method=baseline_method)
    gene_embeddings = interpolate_embeddings(orig_data[0],alphas,
                                              embedding_size=1,
                                              baseline_method=baseline_method)

    num_classes = int(model_emb.output.shape[1])
    gene_matrix = np.zeros([num_classes,orig_data[0].shape[1]])
    drug_matrix = np.zeros([num_classes,example.shape[1],example.shape[2]])
    for i in range(num_classes):
        gradients = compute_gradients_model(model_emb,[gene_embeddings,drug_embeddings],i)
        ig_gene = integral_approximation(gradients=gradients[0]).numpy()
        ig_drug = integral_approximation(gradients=gradients[1]).numpy()
        drug_matrix[i] = ig_drug
        gene_matrix[i] = ig_gene
    return (gene_matrix, drug_matrix)
    
# orig_data = [drug_data,zeros,gene_data]
def get_attributions_paccmann(orig_data, model_con, model_emb,alphas,
                     baseline_method='zeros'):
    # get embedding size for drug tokens
    embedding_size = model_con.outputs[0].shape[2]
    example = model_con(orig_data)
    drug_embeddings = interpolate_embeddings(example.numpy(),alphas,
                                              embedding_size=embedding_size,
                                              baseline_method=baseline_method)
    #print(drug_embeddings.shape)
    gene_embeddings = interpolate_embeddings(orig_data[2],alphas,
                                              embedding_size=1,
                                              baseline_method=baseline_method)
    #print(gene_embeddings.shape)
    
    zeros = np.zeros([gene_embeddings.shape[0],orig_data[1].shape[1]])
    #print(zeros.shape)
    
    num_classes = int(model_emb.output.shape[1])
    gene_matrix = np.zeros([num_classes,orig_data[2].shape[1]])
    drug_matrix = np.zeros([num_classes,example.shape[1],example.shape[2]])
    #print(gene_matrix.shape)
    #print(drug_matrix.shape)
    
    for i in range(num_classes):
        gradients = compute_gradients_model_paccmann(model_emb,[drug_embeddings,zeros,gene_embeddings],i)
        ig_gene = integral_approximation(gradients=gradients[1]).numpy()
        ig_drug = integral_approximation(gradients=gradients[0]).numpy()
        drug_matrix[i] = ig_drug
        gene_matrix[i] = ig_gene
    return (gene_matrix, drug_matrix)
    
def get_attributions_for_batch(batch_data, model_con, model_emb,alphas,
                     baseline_method='zeros'):
    # get embedding size for drug tokens
    embedding_size = model_con.outputs[0].shape[2]
    num_samples = batch_data[0].shape[0]
    for j in range(num_samples):
        orig_data = [batch_data[0][j:j+1],batch_data[1][j:j+1]]
        example = model_con(orig_data)
        drug_embeddings = interpolate_embeddings(example.numpy(),alphas,
                                                  embedding_size=embedding_size,
                                                  baseline_method=baseline_method)
        gene_embeddings = interpolate_embeddings(orig_data[0],alphas,
                                                  embedding_size=1,
                                                  baseline_method=baseline_method)

        num_classes = int(model_emb.output.shape[1])
        if j == 0:
            gene_matrix = np.zeros([num_samples,num_classes,orig_data[0].shape[1]])
            drug_matrix = np.zeros([num_samples,num_classes,example.shape[1],example.shape[2]])
        for i in range(num_classes):
            gradients = compute_gradients_model(model_emb,[gene_embeddings,drug_embeddings],i)
            ig_gene = integral_approximation(gradients=gradients[0]).numpy()
            ig_drug = integral_approximation(gradients=gradients[1]).numpy()
            drug_matrix[j,i] = ig_drug
            gene_matrix[j,i] = ig_gene
        
    return (gene_matrix, drug_matrix)

def min_max_scale_ig(values,use_max=1,use_min=0):
    # min max scaling
    min_val = np.min(values)
    max_val = np.max(values)
    abs_sums_std = (values - min_val) / (max_val - min_val)
    abs_sums_scaled = abs_sums_std * (use_max - use_min) + use_min
    return abs_sums_scaled

def min_max_scale_ig_matrix(value_matrix, use_max = 1, use_min = 0):
    out_matrix = np.zeros(shape = value_matrix.shape)
    for i in range(value_matrix.shape[0]):
        out_matrix[i] = min_max_scale_ig(value_matrix[i],
                        use_max = use_max,
                        use_min = use_min)
    return out_matrix
    
def get_sub_models(model_pre,model_emb, drug_embedding_layer_name = 'drug_embedding'):    
    model_con = tf.keras.Model(inputs=model_pre.inputs, outputs=model_pre.get_layer(drug_embedding_layer_name).output)

    layers = model_pre.layers
    for layer in layers:
        try:
            model_emb.get_layer(layer.name)
        except:
            print('could not find layer : ' + str(layer.name))
            continue
        model_emb.get_layer(layer.name).set_weights(model_pre.get_layer(layer.name).get_weights())


    return model_pre, model_con, model_emb



def plot_drug_importances(drug_importances,
                            drug_vec,
                            character_smiles_dict,
                            batch_mode = False):
    mean_importances = np.mean(drug_importances,axis=0)
    if batch_mode:
        std_importances = np.std(drug_importances,axis=0)
    else:
        std_importances = np.zeros([len(mean_importances),1])
        

    drug_smiles = ''
    drug_imp = []
    drug_std = []
    for i in range(len(drug_vec)):
        char = drug_vec[i]
        try:
            drug_smiles += character_smiles_dict[int(char)]
            drug_imp.append(mean_importances[i])
            if batch_mode:
                drug_std.append(std_importances[i])
            else:
                drug_std.append(0)
        except:
            pass
    smiles_char_list = [a for a in drug_smiles]

    inverse_smiles_char_list = smiles_char_list[::-1]
    inverse_drug_imp = drug_imp[::-1]
    inverse_std_imp = drug_std[::-1]

    plt.figure(figsize=(20,20))
    plt.title("Feature importances")
    plt.barh(range(len(inverse_smiles_char_list)), inverse_drug_imp,
           color="r", xerr=inverse_std_imp, align="center")
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    plt.yticks(range(len(inverse_smiles_char_list)), inverse_smiles_char_list)
    #plt.ylim([-1, X.shape[1]])
    plt.show()

def plot_drug_importances_for_instance(cur_data,
                            model_con, model_emb,alphas,
                            character_smiles_dict):
    gene_matrix, drug_matrix = get_attributions(cur_data,
                              model_con, model_emb,alphas)
    # select the integrated gradients (only one class)
    drug_ig = drug_matrix[0]
    drug_importances = np.sum(np.abs(drug_ig),axis=1)
    drug_importances_scaled = min_max_scale_ig(drug_importances)
    drug_vec = cur_data[1][0]
    
    
    plot_drug_importances(drug_importances_scaled,
                            drug_vec,
                            character_smiles_dict,                            
                            batch_mode = False)
    

def plot_drug_importances_for_batch(batch_data,
                            model_con, model_emb,alphas,
                            character_smiles_dict):
    gene_matrix, drug_matrix = get_attributions_for_batch(batch_data,
                          model_con, model_emb,alphas)
    # select the integrated gradients (only one class)
    drug_ig = drug_matrix[:,0,:,:]
    drug_importances = np.sum(np.abs(drug_ig),axis=2)
    drug_importances_scaled = min_max_scale_ig_matrix(drug_importances)

    drug_vec = batch_data[1][0]
    plot_drug_importances(drug_importances_scaled,
                            drug_vec,
                            character_smiles_dict,                            
                            batch_mode = True)
    
def plot_gene_importances(gene_importances,
                            gene_list,
                            max_show = 30,
                            batch_mode = False):
                            
    mean_importances = np.mean(gene_importances,axis=0)
    if batch_mode:
        std_importances = np.std(gene_importances,axis=0)
    else:
        std_importances = np.zeros([len(mean_importances),])
    
    sort_ids = np.argsort(mean_importances)[::-1]
    show_genes = []
    show_imp   = []
    show_std   = []
    for i in range(np.min([len(sort_ids),max_show])):
        cur_id = sort_ids[i]
        show_genes.append(gene_list[cur_id])
        show_imp.append(mean_importances[cur_id])
        if batch_mode:
            show_std.append(std_importances[cur_id])
        else:
            show_std.append(0)
    
    show_genes = show_genes[::-1]
    show_imp = show_imp[::-1]
    show_std = show_std[::-1]

    plt.figure(figsize=(20,20))
    plt.title("Feature importances")
    plt.barh(range(len(show_genes)), show_imp,
           color="r", xerr=show_std, align="center")
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    plt.yticks(range(len(show_genes)), show_genes)
    #plt.ylim([-1, X.shape[1]])
    plt.show()    



def plot_gene_importances_boxplot(gene_importances_list,
                                gene_list,
                                max_show = 30,
                                group_list = None,
                                plot_swarm = False,
                                figsize = (15,10)):
    mean_importances = np.median(gene_importances_list[0],axis=0)
    sort_ids = np.argsort(mean_importances)[::-1]
    use_ids = sort_ids[0:max_show]
    best_importances_list = []
    for i in range(len(gene_importances_list)):
        best_importances_list.append(gene_importances_list[i][:,use_ids])
    best_genes = np.array(gene_list)[use_ids]
    
    
    df_genes = []
    df_attributions = []
    df_groups = []
    for g_i in range(len(best_importances_list)):
        best_importances = best_importances_list[g_i]
        if group_list is None:
            cur_group = g_i
        else:
            cur_group = group_list[g_i]
        for i in range(best_importances.shape[0]):
            for j in range(best_importances.shape[1]):            
                cur_gene = best_genes[j]        
                cur_val = best_importances[i,j]
                df_genes.append(cur_gene)
                df_attributions.append(cur_val)
                df_groups.append(cur_group)
    
    
    df = pd.DataFrame({'Gene':df_genes,
                       'IG attribution':df_attributions,
                       'Group':df_groups})
    
    plt.figure(figsize=figsize)
    sns.boxplot(x='Gene', y='IG attribution', hue='Group', data = df)
    if plot_swarm:
        sns.swarmplot(x='Gene', y='IG attribution', hue='Group', data = df)
    plt.xticks(rotation=90)
    plt.show()
    
    
    
def plot_gene_importances_for_instance_from_model(cur_data,
                            model_con, model_emb,alphas,
                            gene_list, max_show = 30):
    gene_matrix, drug_matrix = get_attributions(cur_data,
                              model_con, model_emb,alphas)
    # select the integrated gradients (only one class)
    gene_ig = np.abs(gene_matrix[0])
    gene_importances = min_max_scale_ig(gene_ig)
    plot_gene_importances(gene_importances,
                                        gene_list,
                                        max_show = max_show,
                                        batch_mode = False)
    
    
    
def plot_gene_importances_for_batch(batch_data,
                            model_con, model_emb,alphas,
                            gene_list, max_show = 30):
    gene_matrix, drug_matrix = get_attributions_for_batch(batch_data,
                          model_con, model_emb,alphas)

    gene_ig = np.abs(gene_matrix[:,0,:])
    gene_importances = min_max_scale_ig_matrix(gene_ig)
    plot_gene_importances(gene_importances,
                                        gene_list,
                                        max_show = max_show,
                                        batch_mode = True)

    

def get_gene_drug_importances_for_instance_paccmann(cur_data,
                            model_con, model_emb,alphas):
    
    gene_matrix, drug_matrix = get_attributions_paccmann(cur_data,
                              model_con, model_emb,alphas)
    
    # compute scaled gene importances
    # select the integrated gradients (only one class)
    gene_ig = np.abs(gene_matrix[0])
    gene_importances_scaled = min_max_scale_ig(gene_ig)
    
    # compute drug importances
    # select the integrated gradients (only one class)
    drug_ig = drug_matrix[0]
    drug_importances = np.sum(np.abs(drug_ig),axis=1)
    drug_importances_scaled = min_max_scale_ig(drug_importances)
    
    return gene_importances_scaled, drug_importances_scaled

def get_gene_drug_importances_for_instance(cur_data,
                            model_con, model_emb,alphas):
    
    gene_matrix, drug_matrix = get_attributions(cur_data,
                              model_con, model_emb,alphas)
    
    # compute scaled gene importances
    # select the integrated gradients (only one class)
    gene_ig = np.abs(gene_matrix[0])
    gene_importances_scaled = min_max_scale_ig(gene_ig)
    
    # compute drug importances
    # select the integrated gradients (only one class)
    drug_ig = drug_matrix[0]
    drug_importances = np.sum(np.abs(drug_ig),axis=1)
    drug_importances_scaled = min_max_scale_ig(drug_importances)
    
    return gene_importances_scaled, drug_importances_scaled