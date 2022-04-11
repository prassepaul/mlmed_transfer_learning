try:
    import pubchempy as pcp
except:
    print('unable to import pubchempy')
    pass
import requests
import numpy as np
from tqdm import tqdm
import pandas as pd

def load_expression_data(file_path,gene_col,start_offset=2,sep='\t',
                        gene_mapping = None,
                        root_dir = ''):
    data = pd.read_csv(root_dir + file_path,sep=sep)
    symbol_id_mapping = dict()
    gene_expression_dict = dict()
    raw_symbols = list(data[gene_col])
    cell_lines = list(data.keys())[start_offset:]
    raw_data = np.array(data.values[:,start_offset:],dtype=np.float32)
    out_symbols = []
    for i in range(len(raw_symbols)):
        if gene_mapping is not None:
            try:
                cur_symbol = gene_mapping[raw_symbols[i]]
            except:
                continue
        else:
            cur_symbol = raw_symbols[i]
        symbol_id_mapping[cur_symbol] = i
        gene_expression_dict[cur_symbol] = raw_data[i,:]
        out_symbols.append(cur_symbol)
    out_dict = {'symbol_id_mapping':symbol_id_mapping,
                'gene_expression_dict':gene_expression_dict,
                'cell_lines':cell_lines,
                'raw_symbols':out_symbols}
    return out_dict


def create_df_for_gene_list(expression_dict,gene_list,verbose=0):
    if verbose != 0:
        disable = False
    else:
        disable = True
        
    cell_lines = expression_dict['cell_lines']
    raw_symbols = expression_dict['raw_symbols']
    gene_expression_dict = expression_dict['gene_expression_dict']
    gene_data = np.ones([len(gene_list),
                         len(cell_lines)])
    for i in tqdm(np.arange(len(gene_list)), disable = disable):
        if gene_list[i] in gene_expression_dict:
            gene_data[i,:] = gene_expression_dict[gene_list[i]]
    
    raw_df = pd.DataFrame(gene_data)
    raw_df.columns = cell_lines
    raw_df.index = gene_list
    return raw_df


def convert_to_save_df(model_params,metric_dict):
    df_dict = model_params.copy()
    df_dict.update(metric_dict)
    for key in df_dict.keys():
        df_dict[key] = [str(df_dict[key])]
    #print(df_dict)
    return pd.DataFrame(df_dict)

def get_prop_dict_for_compound(compound_name):    

    # search compound on PubChem
    compounds = pcp.get_compounds(compound_name, 'name')

    if(len(compounds)>1):
        #print("Compound: " + compound_name + " Warning: " + str(len(compounds)) + " results found for query on PubChem")
        #print("taking first compound found")
        inchi = compounds[0].to_dict(properties=["inchi"])["inchi"]

    elif len(compounds)==0:
        inchi = None
        return None

    elif len(compounds) == 1:
            #print("Compound: " + compound_name + " found for query on PubChem")
            inchi = compounds[0].to_dict(properties=["inchi"])["inchi"]
    
    return compounds[0].to_dict()
    """
    print(compounds[0].to_dict())
    
    # retrieve InChiKey from chemspider
    if inchi is not None:
        host = "http://www.chemspider.com"
        getstring = "/InChI.asmx/InChIToInChIKey?inchi="

        r = requests.get('{}{}{}'.format(host, getstring, inchi))
        if r.ok:
            inchikey = str(r.text.replace('<?xml version="1.0" encoding="utf-8"?>\r\n<string xmlns="http://www.chemspider.com/">', '').replace('</string>', '').strip())
        else:
             inchikey = None
                
        host = "http://www.chemspider.com"
        getstring = "/InChI.asmx/InChIToSMILES?inchi="

        r = requests.get('{}{}{}'.format(host, getstring, inchi))
        if r.ok:
            smiles = str(r.text.replace('<?xml version="1.0" encoding="utf-8"?>\r\n<string xmlns="http://www.chemspider.com/">', '').replace('</string>', '').strip())
        else:
             smiles = None
    else:
        inchikey = None
        smiles = None

    return inchi, inchikey, smiles
    """
    
def transformation(df, epsilon = 0.0001):
    df = np.arcsinh(df)
    return (df-df.mean())/ (df.std() + epsilon)

def transformation_np(in_matrix, epsilon = 0.0001):
    in_matrix = np.arcsinh(in_matrix)
    return (in_matrix - np.mean(in_matrix,axis=0)) / (np.std(in_matrix,axis=0) + epsilon)
    
def get_gene_list_for_network_prop_df(data_frame_path,
    num_genes_per_drug = 10,
    min_weight_gene = None):
    # read data frame
    in_df = pd.read_csv(data_frame_path,sep='\t')
    
    # collect the prop weights for all the drugs and the genes
    drugs = list(in_df['drug'])
    genes = list(in_df['node'])
    weights = list(in_df['prop_weight'])

    drug_gene_dict = dict()
    for i in tqdm(np.arange(len(drugs))):
        cur_drug = drugs[i]
        cur_gene = genes[i]
        cur_weight = weights[i]
        if cur_drug not in drug_gene_dict:
            drug_gene_dict[cur_drug] = dict()
        drug_gene_dict[cur_drug][cur_gene] = cur_weight
    
    
    
    # create list of genes to use by using the top 
    # <num_genes_per_drug> most important genes 
    # per drug higher <min_weight_gene>
    genes_use = []
    for drug in drug_gene_dict.keys():
        cur_drug_dict = drug_gene_dict[drug]
        gene_list = list(cur_drug_dict.keys())
        prop_list = list(cur_drug_dict.values())
        # sort by prop_weight
        sort_ids = np.argsort(prop_list)[::-1]
        if num_genes_per_drug is not None:
            sort_ids = sort_ids[0:num_genes_per_drug]
        for j in range(len(sort_ids)):
            cur_gene = gene_list[sort_ids[j]]
            cur_val = prop_list[sort_ids[j]]
            if min_weight_gene is not None:
                if cur_val >= min_weight_gene:
                    genes_use.append(cur_gene)
            else:
                genes_use.append(cur_gene)
    genes_use = list(set(genes_use))
    return genes_use